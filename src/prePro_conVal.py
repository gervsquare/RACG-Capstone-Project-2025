def convert_val(d):
    # Find which features have NaN values
    nan_columns = []
    for col in d.columns:
        if d[col].hasnans:
            nan_columns.append(col)
    return nan_columns
    #print("Columns with NaNs:", nan_columns)
