from src.features import FEATURE_COLUMNS, extract_url_features


def test_feature_schema_is_stable():
    row = extract_url_features("https://login.example.com/account/verify?id=123&next=home")
    assert list(row.keys()) == FEATURE_COLUMNS
    assert row["IsHTTPS"] == 1
    assert row["NoOfSubDomain"] == 1
    assert row["NoOfQMarkInURL"] == 1
    assert row["NoOfAmpersandInURL"] == 1


def test_ip_hostname():
    row = extract_url_features("http://192.168.1.10/login")
    assert row["IsDomainIP"] == 1
