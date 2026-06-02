from json_extract_polars import extract_json
import json

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

print("=== Polars Ancestor Explosion ('tags.code.category') ===")
meta, df = extract_json(
    data, 
    explode_paths=["tags.code.category"],
    remove_duplicates=True,
    simplify_columns=True
)
print(df)
