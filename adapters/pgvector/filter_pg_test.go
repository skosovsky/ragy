//go:build integration

package pgvector

import (
	"context"
	"encoding/csv"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"reflect"
	"slices"
	"strconv"
	"strings"
	"testing"
	"time"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/access"
	"github.com/skosovsky/ragy/contracttest"
	"github.com/skosovsky/ragy/dense"
	"github.com/skosovsky/ragy/filter"
	"github.com/skosovsky/ragy/retrieval"
)

type parityMeta struct{ Attributes filter.RawAttributes }
type parityCodec struct{ schema filter.Schema }

func (c parityCodec) Encode(meta parityMeta) (filter.RawAttributes, error) {
	return c.schema.NormalizeAttributes(meta.Attributes)
}
func (c parityCodec) Decode(attrs filter.RawAttributes) (parityMeta, error) {
	normalized, err := c.schema.NormalizeAttributes(attrs)
	return parityMeta{Attributes: normalized}, err
}

// psqlDB is a host test bridge. PostgreSQL prepares the unchanged adapter SQL and
// infers native parameter types. EXECUTE literals are escaped solely by this host
// transport; adapter rendering still uses numbered bound parameters. Exec appends
// RETURNING id inside a CTE to observe actual affected IDs rather than infer them.
type psqlDB struct {
	container string
	affected  []string
	calls     int
}

func (db *psqlDB) command(ctx context.Context, script string) ([][]string, error) {
	cmd := exec.CommandContext(
		ctx,
		"docker",
		"exec",
		"-i",
		db.container,
		"psql",
		"-X",
		"-U",
		"postgres",
		"-d",
		"ragy",
		"-q",
		"-t",
		"--csv",
		"-v",
		"ON_ERROR_STOP=1",
	)
	cmd.Stdin = strings.NewReader("SET standard_conforming_strings=on;\n" + script)
	output, err := cmd.CombinedOutput()
	if err != nil {
		return nil, fmt.Errorf("psql: %w: %s", err, output)
	}
	rows, err := csv.NewReader(strings.NewReader(string(output))).ReadAll()
	return rows, err
}
func preparedScript(query string, args []any) (string, error) {
	literals := make([]string, len(args))
	for i, arg := range args {
		literal, err := parameterLiteral(arg)
		if err != nil {
			return "", err
		}
		literals[i] = literal
	}
	if len(args) == 0 {
		return query + ";", nil
	}
	return "PREPARE ragy_parity AS " + query + ";\nEXECUTE ragy_parity(" + strings.Join(literals, ",") + ");", nil
}
func parameterLiteral(value any) (string, error) {
	switch v := value.(type) {
	case string:
		return "'" + strings.ReplaceAll(v, "'", "''") + "'", nil
	case []byte:
		return parameterLiteral(string(v))
	case []float32:
		data, err := json.Marshal(v)
		if err != nil {
			return "", err
		}
		return parameterLiteral(string(data))
	case int:
		return strconv.Itoa(v), nil
	case int64:
		return strconv.FormatInt(v, 10), nil
	case float64:
		return strconv.FormatFloat(v, 'g', -1, 64), nil
	case bool:
		return strconv.FormatBool(v), nil
	default:
		return "", fmt.Errorf("unsupported test parameter %T", value)
	}
}
func (db *psqlDB) Query(ctx context.Context, query string, args ...any) (Rows, error) {
	db.calls++
	script, err := preparedScript(query, args)
	if err != nil {
		return nil, err
	}
	rows, err := db.command(ctx, script)
	if err != nil {
		return nil, err
	}
	return &psqlRows{rows: rows, index: -1}, nil
}
func (db *psqlDB) Exec(ctx context.Context, query string, args ...any) (Result, error) {
	db.calls++
	script, err := preparedScript("WITH affected AS ("+query+" RETURNING id) SELECT id FROM affected ORDER BY id", args)
	if err != nil {
		return nil, err
	}
	rows, err := db.command(ctx, script)
	if err != nil {
		return nil, err
	}
	db.affected = nil
	for _, row := range rows {
		if len(row) != 1 {
			return nil, errors.New("affected row shape")
		}
		db.affected = append(db.affected, row[0])
	}
	return affectedResult(len(db.affected)), nil
}

type affectedResult int64

func (r affectedResult) RowsAffected() int64 { return int64(r) }

type psqlRows struct {
	rows  [][]string
	index int
}

func (r *psqlRows) Next() bool   { r.index++; return r.index < len(r.rows) }
func (r *psqlRows) Err() error   { return nil }
func (r *psqlRows) Close() error { return nil }
func (r *psqlRows) Scan(dest ...any) error {
	row := r.rows[r.index]
	if len(dest) != len(row) {
		return errors.New("query row shape")
	}
	for i, target := range dest {
		switch v := target.(type) {
		case *string:
			*v = row[i]
		case *[]byte:
			*v = []byte(row[i])
		case *float64:
			value, err := strconv.ParseFloat(row[i], 64)
			if err != nil {
				return err
			}
			*v = value
		default:
			return fmt.Errorf("unsupported test scan %T", target)
		}
	}
	return nil
}

func TestIntegrationPostgresPortableQueryAndDeleteParity(t *testing.T) {
	// Arrange: explicitly isolated test container, process-unique table, complete corpus.
	db := newPostgresDB(t)
	table := fmt.Sprintf("ragy_t09_%d", os.Getpid())
	if _, err := db.command(
		t.Context(),
		"CREATE TABLE "+table+" (id text PRIMARY KEY, content text NOT NULL, attributes jsonb NOT NULL, vector vector(1) NOT NULL);",
	); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		if _, e := db.command(ctx, "DROP TABLE "+table+";"); e != nil {
			t.Error(e)
		}
	})
	versions, err := db.command(
		t.Context(),
		"SELECT current_setting('server_version'), extversion FROM pg_extension WHERE extname='vector';",
	)
	if err != nil || len(versions) != 1 {
		t.Fatal(versions, err)
	}
	t.Logf("PostgreSQL/pgvector %v; table=%s", versions[0], table)
	fixture := contracttest.PortableFilterParity(t)
	store, err := New(
		db,
		Config[parityMeta]{Table: table, Schema: fixture.Schema, Space: fixtureSpace()},
		parityCodec{schema: fixture.Schema},
	)
	if err != nil {
		t.Fatal(err)
	}
	records := make([]dense.Record[parityMeta], len(fixture.Records))
	for i, record := range fixture.Records {
		records[i] = dense.Record[parityMeta]{
			ID:      record.ID,
			Content: "original",
			Meta:    parityMeta{Attributes: record.Attributes},
			Vector:  []float32{1},
			Space:   fixtureSpace(),
		}
	}
	for _, test := range fixture.Cases {
		t.Run(test.Name, func(t *testing.T) { checkPGParityCase(t, db, store, table, fixture, records, test) })
	}
	before := db.calls
	for _, attrs := range fixture.Invalid {
		bad := records[0]
		bad.Meta.Attributes = attrs
		if err = store.Upsert(t.Context(), []dense.Record[parityMeta]{bad}); !errors.Is(err, ragy.ErrInvalidArgument) {
			t.Fatal("malformed metadata admitted", attrs, err)
		}
	}
	if db.calls != before {
		t.Fatal("malformed metadata reached DB")
	}
	t.Logf(
		"corpus=%d predicates=%d malformed=%d; query/delete parity verified",
		len(records),
		len(fixture.Cases),
		len(fixture.Invalid),
	)
}

func checkPGParityCase(
	t *testing.T,
	db *psqlDB,
	store *Store[parityMeta],
	table string,
	fixture contracttest.FilterParityFixture,
	records []dense.Record[parityMeta],
	test contracttest.FilterParityCase,
) {
	t.Helper()
	// Arrange: fresh admitted corpus for each destructive assertion.
	if _, err := db.command(t.Context(), "TRUNCATE "+table+";"); err != nil {
		t.Fatal(err)
	}
	if err := store.Upsert(t.Context(), records); err != nil {
		t.Fatal(err)
	}
	var expected []string
	for _, record := range fixture.Records {
		matched, err := filter.MatchCondition(
			test.Condition,
			func(field string) (any, bool) { value, ok := record.Attributes[field]; return value, ok },
		)
		if err != nil {
			t.Fatal(err)
		}
		if matched {
			expected = append(expected, record.ID)
		}
	}
	slices.Sort(expected)
	// Act: actual adapter query, then actual adapter deletion; no SQL simulation.
	result, err := store.Retrieve(
		t.Context(),
		retrieval.Query[struct{}]{
			Read: access.Unrestricted(),
			Options: retrieval.RetrieveOptions{
				Filters: test.Condition,
				TopK:    len(records),
				Vector:  []float32{1},
				Space:   fixtureSpace(),
			},
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	actual := pgResultIDs(t, result, fixture.Records)

	// Assert: query IDs exactly match the portable matcher, including omissions.
	if !slices.Equal(expected, actual) {
		t.Fatal("query parity", expected, actual)
	}
	deleted, deleteErr := store.DeleteByFilter(t.Context(), test.Condition)
	if test.Name == "empty" {
		if !errors.Is(deleteErr, ragy.ErrInvalidArgument) || deleted.Deleted != 0 {
			t.Fatal("empty delete must fail", deleted, deleteErr)
		}
		return
	}
	if deleteErr != nil || deleted.Deleted != len(expected) || !slices.Equal(db.affected, expected) {
		t.Fatal("delete parity", expected, db.affected, deleted, deleteErr)
	}
	assertPGRemaining(t, db, table, fixture.Records, expected)
}

var _ io.Closer = (*psqlRows)(nil)

func pgResultIDs(
	t *testing.T,
	result retrieval.ResultSet[parityMeta],
	records []contracttest.FilterParityRecord,
) []string {
	t.Helper()
	original := make(map[string]filter.RawAttributes, len(records))
	for _, record := range records {
		original[record.ID] = record.Attributes
	}
	var actual []string
	for _, doc := range result.Documents() {
		attrs, exists := original[doc.ID]
		if !exists || !reflect.DeepEqual(doc.Meta.Attributes, attrs) {
			t.Fatal("metadata integer/scalar roundtrip", doc, attrs)
		}
		actual = append(actual, doc.ID)
	}
	slices.Sort(actual)
	return actual
}

func assertPGRemaining(
	t *testing.T,
	db *psqlDB,
	table string,
	records []contracttest.FilterParityRecord,
	expected []string,
) {
	t.Helper()
	remaining, err := db.command(t.Context(), "SELECT id FROM "+table+" ORDER BY id;")
	if err != nil {
		t.Fatal(err)
	}
	var expectedRemaining []string
	for _, record := range records {
		if !slices.Contains(expected, record.ID) {
			expectedRemaining = append(expectedRemaining, record.ID)
		}
	}
	var remainingIDs []string
	for _, row := range remaining {
		remainingIDs = append(remainingIDs, row[0])
	}
	slices.Sort(expectedRemaining)
	if !slices.Equal(remainingIDs, expectedRemaining) {
		t.Fatal("deleted wrong rows", remainingIDs, expectedRemaining)
	}
}
