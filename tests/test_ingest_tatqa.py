"""Unit tests for the TAT-QA ingestion adapter (P2.2)."""

from src.ingest.tatqa import (
    IngestedTable,
    ingest_context,
    parse_cell,
    split_header_body,
)


# --------------------------------------------------------------------------- #
# cell parsing
# --------------------------------------------------------------------------- #
def test_parse_cell_numbers():
    assert parse_cell("$  1,452.4") == 1452.4
    assert parse_cell("44.1") == 44.1
    assert parse_cell("680") == 680 and isinstance(parse_cell("680"), int)
    assert parse_cell("(94)") == -94          # accounting negative
    assert parse_cell("$ (2,235)") == -2235   # currency before the negative paren
    assert parse_cell("$(0.19)") == -0.19
    assert parse_cell("-12.14") == -12.14
    assert parse_cell("12.5%") == 12.5        # percent stripped


def test_parse_cell_missing_and_text():
    for marker in ("", "  ", "-", "--", "—", "�", "- - %", "n/a", "N/A", "nm"):
        assert parse_cell(marker) == "", f"{marker!r} should normalize to empty"
    assert parse_cell("Automotive") == "Automotive"


# --------------------------------------------------------------------------- #
# header / body splitting
# --------------------------------------------------------------------------- #
def test_split_header_body_multirow_with_superheader():
    grid = [
        ["", "", "Years Ended September 30,", ""],   # merged super-header
        ["", "2019", "2018", "2017"],                # year header
        ["", "", "(in millions)", ""],               # units note
        ["Fixed Price", "$  1,452.4", "$  1,146.2", "$  1,036.9"],
        ["Other", "44.1", "56.7", "70.8"],
    ]
    header, body, start = split_header_body(grid)
    assert start == 3
    assert len(header) == 3 and len(body) == 2
    assert body[0][0] == "Fixed Price"


def test_split_header_body_label_in_col0_year_header():
    # Header row whose col 0 carries a units note and value cells are years.
    grid = [
        ["$ in millions", "2019", "2018"],
        ["Revenue", "100", "90"],
    ]
    header, body, start = split_header_body(grid)
    assert start == 1
    assert body[0][0] == "Revenue"


def test_split_header_body_date_header():
    grid = [
        ["(Dollars in Millions)", "April 27, 2019", "April 28, 2018"],
        ["Revenues", "1,073.3", "1,095"],
    ]
    header, body, start = split_header_body(grid)
    assert start == 1
    assert body[0][0] == "Revenues"


# --------------------------------------------------------------------------- #
# full context ingestion
# --------------------------------------------------------------------------- #
def _ctx(grid):
    return {"table": {"uid": "u1", "table": grid}, "paragraphs": [], "questions": []}


def test_ingest_context_clean_table():
    grid = [
        ["", "", "Fiscal", ""],
        ["", "2019", "2018", "2017"],
        ["", "", "(in millions)", ""],
        ["Automotive", "5,686", "6,092", "5,228"],
        ["Sensors", "914", "918", "814"],
        ["Section header:", "", "", ""],          # dropped (title row)
        ["Total", "13,448", "13,988", "12,185"],
    ]
    it = ingest_context(_ctx(grid), 0)
    assert isinstance(it, IngestedTable)
    assert it.confidence == "high"
    assert it.table.headers == ["category", "2019", "2018", "2017"]
    assert it.metric_cols == ["2019", "2018", "2017"]
    assert it.table.column_types == ["categorical", "numeric", "numeric", "numeric"]
    # section/title rows dropped; only real data rows survive
    assert [r[0] for r in it.table.rows] == ["Automotive", "Sensors", "Total"]
    # values ingested faithfully
    assert it.table.rows[1] == ["Sensors", 914, 918, 814]
    assert "Fiscal" in it.super_headers


def test_ingest_context_drops_stray_section_rows():
    grid = [
        ["", "2019", "2018"],
        ["Revenue", "100", "90"],
        ["", "Costs Data:", ""],                   # stray section row between data
        ["Profit", "40", "30"],
    ]
    it = ingest_context(_ctx(grid), 1)
    assert [r[0] for r in it.table.rows] == ["Revenue", "Profit"]
    assert any("section" in n for n in it.notes)


def test_ingest_disambiguates_duplicate_labels():
    # Multi-section table flattens to repeated labels; each must become unique
    # so a cited/answered row is unambiguous (P3.10 data-quality fix).
    grid = [
        ["", "External", "Internal"],
        ["Leasehold", "23", "77"],
        ["Freehold", "38", "62"],
        ["Leasehold", "60", "40"],
        ["Freehold", "27", "73"],
    ]
    it = ingest_context(_ctx(grid), 0)
    labels = [r[0] for r in it.table.rows]
    assert labels == ["Leasehold", "Freehold", "Leasehold (2)", "Freehold (2)"]
    assert any("disambiguated" in n for n in it.notes)


def test_ingest_drops_midtable_year_section_header():
    # A label-less row carrying only YEARS is a leaked period sub-header, not data.
    grid = [
        ["", "2019", "2018"],
        ["High", "91", "87"],
        ["", "2018", "2018"],   # leaked mid-table period header -> dropped
        ["Low", "70", "75"],
    ]
    it = ingest_context(_ctx(grid), 0)
    assert [r[0] for r in it.table.rows] == ["High", "Low"]


def test_ingest_folds_leaked_year_subheader_into_header():
    # A leading sub-header with years + header text (no real magnitudes) must fold
    # into the header band, not leak in as a data row (years parse as numbers).
    grid = [
        ["(in millions)", "2019", "2018", "Actual", "Comp."],
        ["Orders", "19975", "18451", "8", "7"],
        ["Revenue", "17663", "18125", "-3", "-4"],
    ]
    it = ingest_context(_ctx(grid), 0)
    assert [r[0] for r in it.table.rows] == ["Orders", "Revenue"]


def test_ingested_table_round_trips_schema():
    # Table() construction enforces rectangularity / type-length invariants;
    # a successful ingest means the schema accepted it.
    grid = [["", "2019", "2018"], ["A", "1", "2"], ["B", "3", "4"]]
    it = ingest_context(_ctx(grid), 2)
    assert len(it.table.rows) == 2
    assert all(len(r) == len(it.table.headers) for r in it.table.rows)
