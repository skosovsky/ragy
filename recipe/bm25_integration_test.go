package recipe_test

import (
	"context"
	"testing"

	"github.com/skosovsky/ragy/lexical"
	"github.com/skosovsky/ragy/recipe"
	"github.com/skosovsky/ragy/retrieval"
)

func TestThreeRecipesUseActualScopedBM25(t *testing.T) {
	for _, strategy := range []recipe.Strategy{recipe.SingleRewrite, recipe.MultiQuery, recipe.Decomposition} {
		t.Run(string(strategy), func(t *testing.T) {
			// Arrange: real scoped BM25 corpus; planning/assessment scripted only for contract assertions.
			f := newFixture(t, strategy)
			configureActualBM25(t, f)
			f.assessHook = func(input recipe.AssessmentInput[[]string, []string, meta]) { assertAssessorScope(t, input) }
			switch strategy {
			case recipe.SingleRewrite:
				f.planned, f.selected = []string{"возврат оплаты срок"}, []int{1}
			case recipe.MultiQuery:
				f.planned, f.selected = []string{"сброс пароля", "восстановления доступа"}, []int{1, 2}
			case recipe.Decomposition:
				f.planned, f.selected = []string{"возврат оплаты срок", "банковской картой"}, []int{0, 1}
			}
			// Act.
			result, err := f.run(context.Background(), t)
			// Assert: private corpus entry cannot reach planner assessment, selected evidence or supports.
			if err != nil || result.Outcome != recipe.Complete || len(result.Selected) == 0 {
				t.Fatal(result.Outcome, err)
			}
			assertActualEvidence(t, strategy, result)
		})
	}
}

func configureActualBM25(t *testing.T, f *fixture) {
	t.Helper()
	texts := []string{
		"Возврат оплаты за заказ: срок 10 дней.",
		"Сброс пароля: используйте ссылку восстановления доступа.",
		"Возврат устройства: срок 30 дней.",
		"Оплата банковской картой поддерживается.",
	}
	ids := []string{"d1", "d2", "d3", "d4"}
	docs := make([]retrieval.Document[meta], len(texts)+1)
	for i, text := range texts {
		docs[i] = document(ids[i])
		docs[i].Content = text
	}
	private := document("private")
	private.Content, private.Meta.Tenant = texts[0], "b"
	docs[len(texts)] = private
	b := f.config.Backend.(backend)
	index, err := lexical.NewBM25Snapshot(
		context.Background(),
		b.schema,
		lexical.Config[meta]{SearchFields: []string{"content"}},
		f.read,
		docs,
		f.config.CloneMeta,
	)
	if err != nil {
		t.Fatal(err)
	}
	projected := retrieval.ProjectedBackend[[]string, []string, struct{}, retrieval.NoRequestMeta, meta]{
		Next: index,
		AdmissionProject: func(req request) retrieval.Query[struct{}] {
			return retrieval.Query[struct{}]{
				Read:    req.Read,
				Text:    req.Text,
				Intent:  struct{}{},
				Meta:    retrieval.NoRequestMeta{},
				Options: req.Options,
				Plan:    retrieval.ProjectPlannedQuery(req.Plan, struct{}{}),
			}
		},
		Project: func(req request) retrieval.Query[struct{}] {
			f.retrieved = append(f.retrieved, req.EffectiveText())
			return retrieval.Query[struct{}]{
				Read:    req.Read,
				Text:    req.Text,
				Intent:  struct{}{},
				Meta:    retrieval.NoRequestMeta{},
				Options: req.Options,
				Plan:    retrieval.ProjectPlannedQuery(req.Plan, struct{}{}),
			}
		},
	}
	f.config.Backend = projected
	f.config.Admission = func(ctx context.Context, req request) (retrieval.ReadCoverage, error) {
		_, admissionErr := retrieval.PrepareRead(ctx, req, projected)
		return retrieval.CompleteReadCoverage(), admissionErr
	}
}

func assertActualEvidence(t *testing.T, strategy recipe.Strategy, result recipe.Result[meta]) {
	t.Helper()
	found := map[string]bool{}
	for _, evidence := range result.Selected {
		if evidence.Document.Meta.Tenant != "a" || evidence.Document.ID == "private" {
			t.Fatal("private BM25 evidence escaped")
		}
		found[evidence.Document.ID] = true
	}
	if strategy == recipe.MultiQuery && !found["d2"] {
		t.Fatal("actual multi-query lost d2")
	}
	if strategy != recipe.MultiQuery && !found["d1"] {
		t.Fatal("actual recipe lost d1")
	}
	if strategy == recipe.Decomposition && !found["d4"] {
		t.Fatal("actual decomposition lost second subquestion")
	}
}

func assertAssessorScope(t *testing.T, input recipe.AssessmentInput[[]string, []string, meta]) {
	t.Helper()
	for _, query := range input.Queries {
		for _, doc := range query.Documents {
			if doc.Meta.Tenant != "a" || doc.ID == "private" {
				t.Fatal("private BM25 input reached assessor")
			}
		}
	}
}
