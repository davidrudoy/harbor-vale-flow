from src.crews.crew1_plan import check_plan

COLS = ["Date", "Age"]


def test_valid_plan_passes():
    plan = [{"op": "parse_date", "column": "Date"}, {"op": "fill_missing", "column": "Age", "strategy": "median"}]
    assert check_plan(plan, COLS) == []


def test_bad_plans_are_rejected():
    assert check_plan([{"op": "run_code"}], COLS)  # unknown op
    assert check_plan([{"op": "parse_date", "column": "Nope"}], COLS)  # column does not exist
    assert check_plan([{"op": "fill_missing", "column": "Age", "strategy": "constant"}], COLS)  # no value
    assert check_plan({"op": "parse_date"}, COLS)  # not a list
