package pgvector

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/dense"
	"github.com/skosovsky/ragy/documents"
	"github.com/skosovsky/ragy/filter"
)

const fieldsPerRecord = 4

// Rows iterates over query results.
type Rows interface {
	Next() bool
	Scan(dest ...any) error
	Err() error
	Close() error
}

// Result reports rows affected.
type Result interface {
	RowsAffected() int64
}

// DB executes SQL queries.
type DB interface {
	Query(ctx context.Context, sql string, args ...any) (Rows, error)
	Exec(ctx context.Context, sql string, args ...any) (Result, error)
}

// Config configures the store.
type Config struct {
	Table  string
	Schema filter.Schema
}

// Store is a dense pgvector-backed store.
type Store struct {
	db     DB
	table  string
	schema filter.Schema
}

// New constructs a store.
func New(db DB, cfg Config) (*Store, error) {
	if db == nil {
		return nil, fmt.Errorf("%w: pgvector db", ragy.ErrInvalidArgument)
	}

	if err := filter.ValidateSQLIdentifier(cfg.Table); err != nil {
		return nil, err
	}
	if !cfg.Schema.IsFinalized() {
		return nil, fmt.Errorf("%w: pgvector schema", ragy.ErrInvalidArgument)
	}

	return &Store{db: db, table: cfg.Table, schema: cfg.Schema}, nil
}

// Search implements dense.Searcher.
func (s *Store) Search(ctx context.Context, req dense.Request) ([]ragy.Document, error) {
	if err := req.Validate(); err != nil {
		return nil, err
	}
	if err := s.Schema().ValidateSchemaIR(req.Filter); err != nil {
		return nil, err
	}

	sql, args, err := s.renderSearch(req)
	if err != nil {
		return nil, err
	}

	rows, err := s.db.Query(ctx, sql, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var docs []ragy.Document
	for rows.Next() {
		var (
			id        string
			content   string
			attrsJSON []byte
			relevance float64
		)
		if err := rows.Scan(&id, &content, &attrsJSON, &relevance); err != nil {
			return nil, err
		}

		attrs, err := unmarshalAttributes(attrsJSON)
		if err != nil {
			return nil, err
		}

		doc, err := projectDocument(s.schema, id, content, attrs, relevance)
		if err != nil {
			return nil, err
		}

		docs = append(docs, doc)
	}

	if err := rows.Err(); err != nil {
		return nil, err
	}

	if len(docs) == 0 {
		return nil, nil
	}

	return docs, nil
}

func (s *Store) renderSearch(req dense.Request) (string, []any, error) {
	args := []any{req.Vector}
	where := ""
	if req.Filter != nil {
		rendered, renderedArgs, err := renderFilter(s.schema, req.Filter, len(args)+1)
		if err != nil {
			return "", nil, err
		}
		where = " WHERE " + rendered
		args = append(args, renderedArgs...)
	}

	var builder strings.Builder
	builder.WriteString("SELECT id, content, attributes, 1 / (1 + (vector <=> $1)) AS relevance FROM ")
	builder.WriteString(s.table)
	builder.WriteString(where)
	builder.WriteString(" ORDER BY vector <=> $1")

	if req.Page != nil {
		args = append(args, req.Page.Limit, req.Page.Offset)
		_, _ = fmt.Fprintf(&builder, " LIMIT $%d OFFSET $%d", len(args)-1, len(args))
	}

	return builder.String(), args, nil
}

// Upsert implements dense.Index.
func (s *Store) Upsert(ctx context.Context, records []dense.Record) error {
	if len(records) == 0 {
		return nil
	}

	args := make([]any, 0, len(records)*fieldsPerRecord)
	var values []string
	for index, record := range records {
		if err := record.Validate(); err != nil {
			return err
		}

		attrs, err := s.schema.NormalizeAttributes(record.Attributes)
		if err != nil {
			return err
		}

		attrsJSON, err := json.Marshal(attrs)
		if err != nil {
			return err
		}

		base := index*fieldsPerRecord + 1
		values = append(
			values,
			fmt.Sprintf(
				"($%d,$%d,$%d,$%d)",
				base,
				base+1,
				base+2,
				base+fieldsPerRecord-1,
			),
		)
		args = append(args, record.ID, record.Content, attrsJSON, record.Vector)
	}

	sql := fmt.Sprintf(
		"INSERT INTO %s (id, content, attributes, vector) VALUES %s "+
			"ON CONFLICT (id) DO UPDATE SET content = EXCLUDED.content, attributes = EXCLUDED.attributes, vector = EXCLUDED.vector",
		s.table,
		strings.Join(values, ","),
	)

	_, err := s.db.Exec(ctx, sql, args...)
	return err
}

// FindByIDs implements documents.Store.
func (s *Store) FindByIDs(ctx context.Context, ids []string) ([]ragy.Document, error) {
	if len(ids) == 0 {
		return nil, nil
	}

	placeholders, args := buildIDArgs(ids)
	rows, err := s.db.Query(
		ctx,
		fmt.Sprintf("SELECT id, content, attributes FROM %s WHERE id IN (%s)", s.table, placeholders),
		args...,
	)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var docs []ragy.Document
	for rows.Next() {
		var id, content string
		var attrsJSON []byte
		if err := rows.Scan(&id, &content, &attrsJSON); err != nil {
			return nil, err
		}

		attrs, err := unmarshalAttributes(attrsJSON)
		if err != nil {
			return nil, err
		}

		doc, err := projectDocument(s.schema, id, content, attrs, 0)
		if err != nil {
			return nil, err
		}

		docs = append(docs, doc)
	}

	if err := rows.Err(); err != nil {
		return nil, err
	}

	if len(docs) == 0 {
		return nil, nil
	}

	return docs, nil
}

// DeleteByIDs implements documents.Store.
func (s *Store) DeleteByIDs(ctx context.Context, ids []string) (documents.DeleteResult, error) {
	if len(ids) == 0 {
		return documents.DeleteResult{}, nil
	}

	placeholders, args := buildIDArgs(ids)
	result, err := s.db.Exec(ctx, fmt.Sprintf("DELETE FROM %s WHERE id IN (%s)", s.table, placeholders), args...)
	if err != nil {
		return documents.DeleteResult{}, err
	}

	return documents.DeleteResult{Deleted: int(result.RowsAffected())}, nil
}

// DeleteByFilter implements documents.Store.
func (s *Store) DeleteByFilter(ctx context.Context, expr filter.IR) (documents.DeleteResult, error) {
	if expr == nil {
		return documents.DeleteResult{}, fmt.Errorf("%w: delete filter", ragy.ErrInvalidArgument)
	}
	if filter.IsEmpty(expr) {
		return documents.DeleteResult{}, fmt.Errorf("%w: delete filter", ragy.ErrInvalidArgument)
	}
	if err := s.Schema().ValidateSchemaIR(expr); err != nil {
		return documents.DeleteResult{}, err
	}

	where, args, err := renderFilter(s.schema, expr, 1)
	if err != nil {
		return documents.DeleteResult{}, err
	}

	result, err := s.db.Exec(ctx, fmt.Sprintf("DELETE FROM %s WHERE %s", s.table, where), args...)
	if err != nil {
		return documents.DeleteResult{}, err
	}

	return documents.DeleteResult{Deleted: int(result.RowsAffected())}, nil
}

// Schema returns the finalized filter schema used by the store.
func (s *Store) Schema() filter.Schema {
	return s.schema
}

func renderFilter(schema filter.Schema, expr filter.IR, argStart int) (string, []any, error) {
	walker := &sqlFilterWalker{
		schema:  schema,
		nextArg: argStart,
		stack:   nil,
		result:  sqlRendered{sql: "", args: nil},
	}
	if err := filter.Walk(expr, walker); err != nil {
		return "", nil, err
	}

	return walker.result.sql, walker.result.args, nil
}

func renderScalarComparison(schema filter.Schema, field, op string, value any, argStart int) (string, []any, error) {
	renderedField, err := fieldExpr(schema, field)
	if err != nil {
		return "", nil, err
	}

	return fmt.Sprintf("%s %s $%d", renderedField, op, argStart), []any{value}, nil
}

func renderMembership(
	schema filter.Schema,
	fieldName string,
	values []filter.Value,
	argStart int,
) (string, []any, error) {
	field, err := fieldExpr(schema, fieldName)
	if err != nil {
		return "", nil, err
	}

	args := make([]any, 0, len(values))
	placeholders := make([]string, 0, len(values))
	for index, value := range values {
		args = append(args, value.Raw())
		placeholders = append(placeholders, fmt.Sprintf("$%d", argStart+index))
	}

	return fmt.Sprintf("%s IN (%s)", field, strings.Join(placeholders, ",")), args, nil
}

func buildIDArgs(ids []string) (string, []any) {
	args := make([]any, 0, len(ids))
	placeholders := make([]string, 0, len(ids))
	for index, id := range ids {
		placeholders = append(placeholders, fmt.Sprintf("$%d", index+1))
		args = append(args, id)
	}
	return strings.Join(placeholders, ","), args
}

func unmarshalAttributes(data []byte) (ragy.Attributes, error) {
	if len(data) == 0 {
		return ragy.Attributes{}, nil
	}

	out := make(ragy.Attributes)
	if err := json.Unmarshal(data, &out); err != nil {
		return nil, err
	}

	if len(out) == 0 {
		return ragy.Attributes{}, nil
	}

	return out, nil
}

func projectDocument(
	schema filter.Schema,
	id, content string,
	attrs ragy.Attributes,
	relevance float64,
) (ragy.Document, error) {
	normalized, err := schema.NormalizeAttributes(attrs)
	if err != nil {
		return ragy.Document{}, err
	}

	doc := ragy.Document{
		ID:         id,
		Content:    content,
		Attributes: ragy.CloneAttributes(normalized),
		Relevance:  ragy.ClampRelevance(relevance),
	}
	return ragy.NormalizeDocument(doc)
}

func fieldExpr(schema filter.Schema, field string) (string, error) {
	kind, ok := schema.Lookup(field)
	if !ok {
		return "", fmt.Errorf("%w: undeclared schema field %q", ragy.ErrInvalidArgument, field)
	}

	switch kind {
	case filter.KindString:
		return fmt.Sprintf("attributes->>'%s'", field), nil
	case filter.KindInt:
		return fmt.Sprintf("(attributes->>'%s')::bigint", field), nil
	case filter.KindFloat:
		return fmt.Sprintf("(attributes->>'%s')::double precision", field), nil
	case filter.KindBool:
		return fmt.Sprintf("(attributes->>'%s')::boolean", field), nil
	default:
		return "", fmt.Errorf("%w: unsupported pgvector filter kind %q", ragy.ErrUnsupported, kind)
	}
}

type sqlRendered struct {
	sql  string
	args []any
}

type sqlFrame struct {
	op    string
	items []sqlRendered
}

type sqlFilterWalker struct {
	schema  filter.Schema
	nextArg int
	stack   []sqlFrame
	result  sqlRendered
}

func (w *sqlFilterWalker) OnEmpty() error {
	return w.push(sqlRendered{sql: "TRUE", args: nil})
}

func (w *sqlFilterWalker) OnEq(field string, value filter.Value) error {
	return w.pushScalar(field, "=", value)
}

func (w *sqlFilterWalker) OnNeq(field string, value filter.Value) error {
	return w.pushScalar(field, "<>", value)
}

func (w *sqlFilterWalker) OnGt(field string, value filter.Value) error {
	return w.pushScalar(field, ">", value)
}

func (w *sqlFilterWalker) OnGte(field string, value filter.Value) error {
	return w.pushScalar(field, ">=", value)
}

func (w *sqlFilterWalker) OnLt(field string, value filter.Value) error {
	return w.pushScalar(field, "<", value)
}

func (w *sqlFilterWalker) OnLte(field string, value filter.Value) error {
	return w.pushScalar(field, "<=", value)
}

func (w *sqlFilterWalker) OnIn(field string, values []filter.Value) error {
	rendered, args, err := renderMembership(w.schema, field, values, w.nextArg)
	if err != nil {
		return err
	}
	w.nextArg += len(args)
	return w.push(sqlRendered{sql: rendered, args: args})
}

func (w *sqlFilterWalker) EnterAnd(_ int) error {
	w.stack = append(w.stack, sqlFrame{op: "AND", items: nil})
	return nil
}

func (w *sqlFilterWalker) LeaveAnd() error {
	return w.leaveGroup("AND")
}

func (w *sqlFilterWalker) EnterOr(_ int) error {
	w.stack = append(w.stack, sqlFrame{op: "OR", items: nil})
	return nil
}

func (w *sqlFilterWalker) LeaveOr() error {
	return w.leaveGroup("OR")
}

func (w *sqlFilterWalker) EnterNot() error {
	w.stack = append(w.stack, sqlFrame{op: "NOT", items: nil})
	return nil
}

func (w *sqlFilterWalker) LeaveNot() error {
	frame, err := w.popFrame("NOT")
	if err != nil {
		return err
	}
	if len(frame.items) != 1 {
		return fmt.Errorf("%w: invalid NOT filter", ragy.ErrUnsupported)
	}

	item := frame.items[0]
	return w.push(sqlRendered{
		sql:  fmt.Sprintf("NOT (%s)", item.sql),
		args: append([]any(nil), item.args...),
	})
}

func (w *sqlFilterWalker) pushScalar(field, op string, value filter.Value) error {
	rendered, args, err := renderScalarComparison(w.schema, field, op, value.Raw(), w.nextArg)
	if err != nil {
		return err
	}
	w.nextArg += len(args)
	return w.push(sqlRendered{sql: rendered, args: args})
}

func (w *sqlFilterWalker) leaveGroup(op string) error {
	frame, err := w.popFrame(op)
	if err != nil {
		return err
	}

	parts := make([]string, 0, len(frame.items))
	args := make([]any, 0)
	for _, item := range frame.items {
		parts = append(parts, "("+item.sql+")")
		args = append(args, item.args...)
	}

	return w.push(sqlRendered{
		sql:  strings.Join(parts, " "+op+" "),
		args: args,
	})
}

func (w *sqlFilterWalker) push(item sqlRendered) error {
	if len(w.stack) == 0 {
		w.result = item
		return nil
	}

	last := len(w.stack) - 1
	w.stack[last].items = append(w.stack[last].items, item)
	return nil
}

func (w *sqlFilterWalker) popFrame(op string) (sqlFrame, error) {
	if len(w.stack) == 0 {
		return sqlFrame{}, fmt.Errorf("%w: unmatched %s filter", ragy.ErrUnsupported, op)
	}

	last := len(w.stack) - 1
	frame := w.stack[last]
	w.stack = w.stack[:last]
	if frame.op != op {
		return sqlFrame{}, fmt.Errorf("%w: unexpected filter group %q", ragy.ErrUnsupported, frame.op)
	}

	return frame, nil
}

var (
	_ dense.Searcher  = (*Store)(nil)
	_ dense.Index     = (*Store)(nil)
	_ documents.Store = (*Store)(nil)
)
