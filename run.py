from pipeline_utils_4 import json_to_cohort_df

file_path = "run.json"

static_df, aggr_df, variable_ranges_df, print_statement = json_to_cohort_df(file_path)



print("\n\n\n")

print("aggr_df: ", aggr_df.head())
print("\n\n\n")
print("static_df: ", static_df.head())
print("\n\n\n")
print("variable_ranges_df: ", variable_ranges_df.head())