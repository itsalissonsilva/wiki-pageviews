import pandas as pd

from wiki_pageviews.data import load_panel, parse_page_id, prepare_wide_csv


def test_page_parser_preserves_underscores_in_title():
    parsed = parse_page_id("A_page_with_spaces_en.wikipedia.org_all-access_all-agents")
    assert parsed["page"] == "A_page_with_spaces"
    assert parsed["language"] == "en"
    assert parsed["access"] == "all-access"
    assert parsed["agent"] == "all-agents"


def test_prepare_streams_wide_csv_to_panel(tmp_path):
    source = tmp_path / "train.csv"
    destination = tmp_path / "train.parquet"
    pd.DataFrame(
        {
            "Page": [
                "Alpha_en.wikipedia.org_all-access_all-agents",
                "Beta_fr.wikipedia.org_desktop_all-agents",
            ],
            "2024-01-01": [1, 3],
            "2024-01-02": [2, 4],
        }
    ).to_csv(source, index=False)
    prepare_wide_csv(source, destination, chunksize=1)
    panel = load_panel(destination)
    assert len(panel) == 4
    assert panel["unique_id"].nunique() == 2
    assert set(panel["language"]) == {"en", "fr"}
