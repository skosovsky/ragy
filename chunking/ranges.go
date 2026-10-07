package chunking

import (
	"context"
	"slices"
	"sort"
	"strings"
	"unicode"
	"unicode/utf8"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/source"
)

type rangeTask struct {
	span      source.ByteSpan
	separator int
}

func trimRange(ctx context.Context, text string, span source.ByteSpan) (source.ByteSpan, error) {
	for span.Start < span.End {
		if err := ctx.Err(); err != nil {
			return source.ByteSpan{}, err
		}
		r, width := utf8.DecodeRuneInString(text[span.Start:span.End])
		if !unicode.IsSpace(r) {
			break
		}
		span.Start += width
	}
	for span.Start < span.End {
		if err := ctx.Err(); err != nil {
			return source.ByteSpan{}, err
		}
		r, width := utf8.DecodeLastRuneInString(text[span.Start:span.End])
		if !unicode.IsSpace(r) {
			break
		}
		span.End -= width
	}
	return span, ctx.Err()
}

func runeOffsets(ctx context.Context, text string, span source.ByteSpan) ([]int, error) {
	offsets := []int{}
	for offset := range text[span.Start:span.End] {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		offsets = append(offsets, span.Start+offset)
	}
	return append(offsets, span.End), ctx.Err()
}

func splitRanges(
	ctx context.Context,
	text string,
	span source.ByteSpan,
	size, overlap int,
	separators []string,
) ([]source.ByteSpan, error) {
	offsets, err := runeOffsets(ctx, text, span)
	if err != nil {
		return nil, err
	}
	stack := []rangeTask{{span: span, separator: 0}}
	var out []source.ByteSpan
	for len(stack) > 0 {
		if err = ctx.Err(); err != nil {
			return nil, err
		}
		task := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		trimmed, trimErr := trimRange(ctx, text, task.span)
		if trimErr != nil {
			return nil, trimErr
		}
		if trimmed.Start == trimmed.End {
			continue
		}
		start, end := sort.SearchInts(offsets, trimmed.Start), sort.SearchInts(offsets, trimmed.End)
		if end-start <= size {
			out = append(out, trimmed)
			continue
		}
		if task.separator >= len(separators) {
			pieces, fixedErr := fixedRanges(ctx, text, offsets, start, end, size, overlap)
			if fixedErr != nil {
				return nil, fixedErr
			}
			out = append(out, pieces...)
			continue
		}
		pieces, splitErr := separatorRanges(ctx, text, trimmed, separators[task.separator], size, offsets)
		if splitErr != nil {
			return nil, splitErr
		}
		for _, piece := range slices.Backward(pieces) {
			stack = append(stack, rangeTask{span: piece, separator: task.separator + 1})
		}
	}
	return out, ctx.Err()
}

func separatorRanges(
	ctx context.Context,
	text string,
	span source.ByteSpan,
	separator string,
	size int,
	offsets []int,
) ([]source.ByteSpan, error) {
	var out []source.ByteSpan
	current := source.ByteSpan{Start: 0, End: 0}
	for cursor := span.Start; cursor < span.End; {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		index := strings.Index(text[cursor:span.End], separator)
		end := span.End
		if index >= 0 {
			end = cursor + index
		}
		piece, err := trimRange(ctx, text, source.ByteSpan{Start: cursor, End: end})
		if err != nil {
			return nil, err
		}
		if piece.Start < piece.End {
			switch {
			case current.End == 0:
				current = piece
			case sort.SearchInts(offsets, piece.End)-sort.SearchInts(offsets, current.Start) <= size:
				current.End = piece.End
			default:
				out = append(out, current)
				current = piece
			}
		}
		if index < 0 {
			break
		}
		cursor = end + len(separator)
	}
	if current.Start < current.End {
		out = append(out, current)
	}
	return out, ctx.Err()
}

func markdownRanges(ctx context.Context, text string) ([]source.ByteSpan, error) {
	var out []source.ByteSpan
	start := 0
	for cursor := 0; cursor < len(text); {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		end := len(text)
		if next := strings.IndexByte(text[cursor:], '\n'); next >= 0 {
			end = cursor + next
		}
		if cursor > start && strings.HasPrefix(strings.TrimSpace(text[cursor:end]), "#") {
			out = append(out, source.ByteSpan{Start: start, End: cursor})
			start = cursor
		}
		cursor = end + 1
	}
	if start < len(text) {
		out = append(out, source.ByteSpan{Start: start, End: len(text)})
	}
	return out, ctx.Err()
}

func sentenceTexts(ctx context.Context, text string, spans []source.ByteSpan) ([]string, error) {
	if len(spans) == 0 {
		return nil, ragy.ErrProtocol
	}
	out := make([]string, len(spans))
	previous := 0
	for i, span := range spans {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		if span.Start < previous || span.ValidateText(text) != nil ||
			strings.TrimSpace(text[previous:span.Start]) != "" ||
			strings.TrimSpace(text[span.Start:span.End]) == "" {
			return nil, ragy.ErrProtocol
		}
		out[i] = text[span.Start:span.End]
		previous = span.End
	}
	if strings.TrimSpace(text[previous:]) != "" {
		return nil, ragy.ErrProtocol
	}
	return out, nil
}

func fixedRanges(
	ctx context.Context,
	text string,
	offsets []int,
	start, end, size, overlap int,
) ([]source.ByteSpan, error) {
	var out []source.ByteSpan
	for at := start; at < end; at += size - overlap {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		until := min(at+size, end)
		fragment, err := trimRange(ctx, text, source.ByteSpan{Start: offsets[at], End: offsets[until]})
		if err != nil {
			return nil, err
		}
		if fragment.Start < fragment.End {
			out = append(out, fragment)
		}
		if until == end {
			break
		}
	}
	return out, nil
}
