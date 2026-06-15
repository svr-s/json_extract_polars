import polars as pl
import itertools
import fnmatch
import re
import json
import json5
import yaml
import os
from typing import Union, Optional, Tuple, Dict, Any

def _flatten_and_expand(data, parent_key='', explode_paths=None):
    """
    Recursively flattens a nested dictionary or list and explodes lists 
    into multiple rows via Cartesian product.
    Nested keys are separated by a dot (.).
    For example: parentEntity.ChildEntity1.SubEntity2
    """
    if isinstance(data, dict):
        if not data:
            yield {parent_key: None} if parent_key else {}
            return
            
        key_results = []
        for k, v in data.items():
            new_key = f"{parent_key}.{k}" if parent_key else k
            # Pass the generator itself to itertools.product
            key_results.append(_flatten_and_expand(v, new_key, explode_paths))
            
        for combination in itertools.product(*key_results):
            merged = {}
            for d in combination:
                merged.update(d)
            yield merged
            
    elif isinstance(data, list):
        if not data:
            yield {parent_key: None}
            return
            
        # Determine if we have permission to explode this list
        should_explode = False
        if explode_paths is not None:
            for path in explode_paths:
                # 1. Exact match or wildcard match
                if fnmatch.fnmatch(parent_key, path):
                    should_explode = True
                    break
                # 2. Ancestor explosion: if the requested path is a child of this list
                # e.g. parent_key="tags", path="tags.code.category" or "tags.*"
                if path.startswith(parent_key + "."):
                    should_explode = True
                    break
                    
        if not should_explode:
            # Safely stringify the entire list and prevent the combinatorial explosion
            yield {parent_key: json.dumps(data)}
            return
            
        for item in data:
            yield from _flatten_and_expand(item, parent_key, explode_paths)
        
    else:
        # Base case: primitive value
        yield {parent_key: data}

def extract_json(
    json_data: Union[Dict, list], 
    desired_columns: Optional[list] = None, 
    explode_paths: Optional[list] = None,
    row_filters: Optional[Dict[str, Any]] = None, 
    remove_duplicates: bool = False, 
    simplify_columns: bool = False, 
    remove_empty: Union[bool, str] = False,
    sort_columns: bool = False,
    keep_all_columns: bool = False
) -> Tuple[Dict[str, Any], pl.DataFrame]:
    """
    Takes parsed JSON data, extracts the primary records, flattens deeply nested 
    hierarchies into a polars DataFrame using dot-notation, and explodes lists 
    via Cartesian product.
    
    This function acts as a robust JSON-to-Polars data extraction engine, specifically
    built to handle complex, heavily-nested JSON payloads. It automatically gracefully 
    handles raw primitive arrays, standardizes column names, and allows for powerful 
    wildcard column extraction, list-based row filtering, and intelligent record unpacking 
    for nested batches.
    
    Args:
        json_data (dict | list): 
            The parsed JSON structure to explode. 
            Can be a deeply nested dictionary, a list of dictionaries, or even a 
            raw 2D array (e.g., `[[0, 1], [2, 3]]`) which will be automatically 
            mapped to `col1`, `col2`, etc.
            
        desired_columns (list, optional): 
            A list of column names, indices, or wildcards to retain in the final DataFrame.
            - Example (Exact): `["accountId", "user.profile.firstName"]`
            - Example (Index): `["5", 5]` (Extracts exactly the 5th column, 1-indexed)
            - Example (Range): `["1-7"]` (Extracts columns 1 through 7 inclusive)
            - Example (Prefix Wildcard): `["shippingAddress.*"]`
            - Example (Suffix Wildcard): `["*.statusCode"]`
            If indices are used alongside other matches that result in duplicates (e.g., `["5", "1-7"]`), 
            the final columns are safely deduplicated.
            If a requested column or wildcard matches no columns, a warning is printed.
            
        explode_paths (list, optional):
            A list of specific array paths to explode. By default (None), NO nested lists 
            are exploded into multiple rows; instead, they are serialized to strings to 
            safely prevent Out-Of-Memory (OOM) combinatorial explosions on large payloads.
            If a list path is NOT included in `explode_paths`, the flattener will halt and 
            serialize the entire unexploded list into a string.
            - Example (Exact): `["line_items", "tags"]`
            - Example (Wildcard): `["*.line_items"]`
            
        row_filters (dict, optional): 
            A dictionary mapping column names to desired values to filter the dataset.
            Values can be a single exact match or a list of acceptable matches.
            - Example (Exact Match): `{"accountId": "ACC-99823-XYZ"}`
            - Example (List Match): `{"regionCode": ["US-EAST", "EU-WEST"]}`
            
        remove_duplicates (bool, optional): 
            If True, removes completely identical rows from the final DataFrame 
            after all filtering has been applied. Defaults to False.
            
        simplify_columns (bool, optional): 
            If True, intelligently strips parent prefixes from column names, retaining 
            only the final child property (e.g., `parent.child.shortName` becomes `shortName`).
            If doing so would result in duplicate column names (e.g., both `a.code` and `b.code` 
            become `code`), it safely preserves their full dot-notation names to prevent collisions.
            Defaults to False.
            
        remove_empty (bool | str, optional): 
            Cleans the dataset by dropping rows containing missing values (`NaN`, `None`) 
            or completely empty strings (`""`, `"   "`).
            - `'any'` (or True): Drops the row if *any* of its filtered columns are missing.
            - `'all'`: Drops the row only if *all* of its filtered columns are missing.
            - False: Disables empty row removal. Defaults to False.
            
        sort_columns (bool, optional):
            If True, sorts the columns alphabetically. If used with `desired_columns` 
            and `keep_all_columns=True`, it sorts the "remaining" columns that were 
            not explicitly pinned to the front. Defaults to False.
            
        keep_all_columns (bool, optional):
            Controls the behavior of `desired_columns`.
            - False (default): `desired_columns` acts as a filter. Only requested columns are kept.
            - True: `desired_columns` acts as a priority list. Requested columns appear first 
              in order, followed by all other columns found in the dataset.
    
    Returns:
        tuple: A tuple containing:
            - meta_data_dict (dict): A dictionary describing the resulting table size 
              and the exact schema/column names generated. The `column_names` key contains 
              a dictionary mapping the 1-based numerical index to the respective column name 
              (e.g., `{1: "accountId", 2: "user.profile.firstName"}`).
            - pandas_dataframe (pl.DataFrame): The fully flattened, filtered, and 
              cleaned polars DataFrame.
              
    Raises:
        ValueError: If `json_data` is not a dict or list, or if parameter types are invalid.
        KeyError: If an exact column requested in `desired_columns` or `row_filters` 
                  does not exist in the dataset.
    """
    if not isinstance(json_data, (dict, list)):
        raise ValueError(f"Invalid input: json_data must be a dictionary or a list, got {type(json_data).__name__}.")
        
    # Determine the records to process
    if isinstance(json_data, list):
        records = json_data
    elif isinstance(json_data, dict):
        # Look for keys that contain lists of dictionaries
        list_keys = [k for k, v in json_data.items() if isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict)]
        if list_keys:
            # Pick the key with the largest list, assuming that's the primary data payload
            largest_list_key = max(list_keys, key=lambda k: len(json_data[k]))
            records = json_data[largest_list_key]
        else:
            # Otherwise, treat the entire dictionary as a single record
            records = [json_data]
    else:
        records = [{"value": json_data}]
        
    # Format records
    # Handle nested batches: if a record is a list and contains dictionaries, 
    # we expand it into separate records. This handles structures like [[{obj1}, {obj2}]] 
    # which often appear in paginated or batched API responses.
    final_records = []
    for r in records:
        if isinstance(r, list):
            # If the list contains at least one dictionary, assume it's a batch of records
            if any(isinstance(item, dict) for item in r):
                final_records.extend(r)
            else:
                # Treat as a single row-record with col1, col2... (CSV-like JSON)
                final_records.append({f"col{i+1}": val for i, val in enumerate(r)})
        else:
            final_records.append(r)
    records = final_records
        
    # Flatten and explode each record using lazy evaluation (generator)
    def generate_all_rows():
        for record in records:
            yield from _flatten_and_expand(record, explode_paths=explode_paths)
    
    # Create DataFrame directly from the generator
    # We materialize to list here because Polars from_dicts expects a sequence
    df = pl.from_dicts(list(generate_all_rows()))
    
    # Apply Column Selection and Sorting
    all_dataset_columns = list(df.columns)
    final_columns = all_dataset_columns  # Default to all columns in original order
    
    if desired_columns:
        if not isinstance(desired_columns, list):
            raise ValueError("Invalid input: desired_columns must be a list of strings.")
            
        matched_priority_columns = []
        for req_col in desired_columns:
            req_str = str(req_col).strip()
            
            # Check for range pattern (e.g., "1-7")
            if re.match(r'^\d+-\d+$', req_str):
                start, end = map(int, req_str.split('-'))
                start_idx = max(0, start - 1)
                end_idx = min(len(all_dataset_columns), end)
                matched_cols = all_dataset_columns[start_idx:end_idx]
                if sort_columns:
                    matched_cols.sort()
                if not matched_cols:
                    print(f"Warning: Index range '{req_str}' is entirely out of bounds.")
                matched_priority_columns.extend(matched_cols)
                
            # Check for single digit index (e.g., "5" or 5)
            elif req_str.isdigit():
                idx = int(req_str) - 1
                if 0 <= idx < len(all_dataset_columns):
                    matched_priority_columns.append(all_dataset_columns[idx])
                else:
                    print(f"Warning: Index '{req_str}' is out of bounds.")
                    
            # Check for wildcards
            elif '*' in req_str or '?' in req_str:
                matched_cols = [c for c in all_dataset_columns if fnmatch.fnmatch(c, req_str)]
                if sort_columns:
                    matched_cols.sort()
                if not matched_cols:
                    print(f"Warning: Wildcard filter '{req_str}' did not match any columns.")
                matched_priority_columns.extend(matched_cols)
                
            # Exact match
            else:
                if req_str in all_dataset_columns:
                    matched_priority_columns.append(req_str)
                else:
                    raise KeyError(f"Requested column '{req_str}' does not exist.")
                    
        # Deduplicate priority list while preserving order
        seen = set()
        priority_ordered = [x for x in matched_priority_columns if not (x in seen or seen.add(x))]
        
        if keep_all_columns:
            # Find the rest of the columns not in the priority list
            remaining = [c for c in all_dataset_columns if c not in seen]
            if sort_columns:
                remaining.sort()
            final_columns = priority_ordered + remaining
        else:
            final_columns = priority_ordered
            
    elif sort_columns:
        # No desired_columns provided, just sort everything
        final_columns = sorted(all_dataset_columns)
        
    df = df.select(final_columns)
        
    # Apply Row Filters
    if row_filters:
        if not isinstance(row_filters, dict):
            raise ValueError("Invalid input: row_filters must be a dictionary.")
            
        for col, val in row_filters.items():
            if col in df.columns:
                if isinstance(val, list):
                    df = df.filter(pl.col(col).is_in(val))
                else:
                    df = df.filter(pl.col(col) == val)
            else:
                raise KeyError(f"Row filter column '{col}' does not exist in the exploded dataset.")
                
    # Remove Duplicates
    if remove_duplicates:
        df = df.unique(maintain_order=True)
        
    # Simplify Column Names
    if simplify_columns:
        from collections import defaultdict
        short_to_original = defaultdict(list)
        
        for col in df.columns:
            short_name = col.split('.')[-1]
            short_to_original[short_name].append(col)
            
        rename_map = {}
        for short_name, originals in short_to_original.items():
            if len(originals) == 1:
                # Unique short name, safe to rename
                rename_map[originals[0]] = short_name
            # If len > 1, there is a collision, so we do NOT add them to the rename map.
            # They will naturally retain their full original names.
            
        df = df.rename(rename_map)
        
    # Remove Empty/Blanks/NaNs
    if remove_empty:
        # Convert empty strings or whitespace-only strings to proper NaNs
        for col in df.columns:
            if df[col].dtype == pl.Utf8:
                df = df.with_columns(
                    pl.when(pl.col(col).str.strip_chars() == "").then(None).otherwise(pl.col(col)).alias(col)
                )
        
        if isinstance(remove_empty, str) and remove_empty.lower() == 'all':
            df = df.filter(~pl.all_horizontal(pl.col("*").is_null()))
        else:
            df = df.drop_nulls()
                
    # Generate Metadata
    meta_data = {
        "table_size": {
            "rows": df.height,
            "columns": df.width
        },
        "column_names": {i + 1: col for i, col in enumerate(df.columns)}
    }
    
    return meta_data, df

def extract_file(
    filepath: str, 
    desired_columns: Optional[list] = None, 
    explode_paths: Optional[list] = None,
    row_filters: Optional[Dict[str, Any]] = None, 
    remove_duplicates: bool = False, 
    simplify_columns: bool = False, 
    remove_empty: Union[bool, str] = False,
    sort_columns: bool = False,
    keep_all_columns: bool = False
) -> Tuple[Dict[str, Any], pl.DataFrame]:
    """
    Reads a file (JSON, JSON5, or YAML) from disk, parses it, and pipes it directly into `extract_json`.
    
    Args:
        filepath (str): The path to the `.json`, `.json5`, `.yaml`, or `.yml` file.
        [...other args map directly to extract_json]
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"The file {filepath} does not exist.")
        
    ext = os.path.splitext(filepath)[1].lower()
    
    with open(filepath, 'r', encoding='utf-8') as f:
        if ext in ['.yaml', '.yml']:
            data = yaml.safe_load(f)
        elif ext == '.json5':
            data = json5.load(f)
        else:
            data = json.load(f)
            
    return extract_json(
        json_data=data,
        desired_columns=desired_columns,
        explode_paths=explode_paths,
        row_filters=row_filters,
        remove_duplicates=remove_duplicates,
        simplify_columns=simplify_columns,
        remove_empty=remove_empty,
        sort_columns=sort_columns,
        keep_all_columns=keep_all_columns
    )

if __name__ == '__main__':
    # Localized test case to verify the function works
    import json
    import os
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sample_file = os.path.join(current_dir, 'sample_json.json')
    
    if os.path.exists(sample_file):
        print(f"Loading test data from: {sample_file}")
        with open(sample_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Define some filters for testing
        col_filters = [
            "languageCode", 
            "laborAllocationCodes.code", 
            "sample_test_4.sample_test_4_1"
        ]
        r_filters = {
            "languageCode": "en-CA"
        }
        
        print(f"Applying Column Filters: {col_filters}")
        print(f"Applying Row Filters: {r_filters}")
        
        meta, exploded_df = extract_json(
            json_data=data, 
            desired_columns=col_filters, 
            explode_paths=["laborAllocationCodes"], # Only explode this array, stringify others like sample_test_4
            row_filters=r_filters,
            simplify_columns=False,
            remove_empty='all'
        )
        
        print("\n--- Metadata Result ---")
        print(json.dumps(meta, indent=4))
        
        print("\n--- DataFrame Result ---")
        print(exploded_df)
    else:
        print("Please provide a valid JSON file to test.")
