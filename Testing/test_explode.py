from json_extract_polars import extract_json

def test_ancestor_explosion():
    data = [{
        "orderid": "ORD-123",
        "line_items": [
            {"sku": "L1"}, {"sku": "L2"}
        ],
        "tags": [
            {
                "code": {
                    "category": ["A", "B"]
                },
                "value": ["v1", "v2"]
            },
            {
                "code": {
                    "category": ["C", "D"]
                },
                "value": ["v3", "v4"]
            }
        ]
    }]

    meta, df = extract_json(data, explode_paths=["tags.code.category"])
    
    assert df.height == 4, f"Expected 4 rows, got {df.height}"
    assert df['orderid'][0] == "ORD-123"

def test_no_explosion_default():
    data = [{
        "orderid": "ORD-123",
        "line_items": [
            {"sku": "L1"}, {"sku": "L2"}
        ]
    }]

    meta, df = extract_json(data)
    assert df.height == 1, f"Expected 1 row, got {df.height}"
