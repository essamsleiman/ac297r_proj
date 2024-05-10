


icuids = labs_df['stay_id'].unique()
treatments = ['norepinephrine', 'vasopressin', 'phenylephrine', 'dopamine', 'epinephrine']
vaso_treatments = load_treatments(data_path, icuids, treatments)

# select the vassopressin vaso_rate that are less than or equal to 0.2
vaso_treatments['vasopressin'] = vaso_treatments['vasopressin'][vaso_treatments['vasopressin']['vaso_rate'] <= 0.2]

# multiply the vaso_rate of vasopressin by 5
vaso_treatments['vasopressin']['vaso_rate'] = vaso_treatments['vasopressin']['vaso_rate']*5

# multiply the vaso_rate of phenylephrine by 0.45
vaso_treatments['phenylephrine']['vaso_rate'] = vaso_treatments['phenylephrine']['vaso_rate']*0.45

# multiply the vaso_rate of dopamine by 0.01
vaso_treatments['dopamine']['vaso_rate'] = vaso_treatments['dopamine']['vaso_rate']*0.01

def combine_treatments(vaso_treatments):
    vaso_df = pd.DataFrame()
    for key, value in vaso_treatments.items():
        value['vaso'] = key
        vaso_df = pd.concat([vaso_df, value], axis = 0)
    return vaso_df

vaso_df = combine_treatments(vaso_treatments)

vaso_df_time = vaso_df.sort_values(by = ['stay_id', 'starttime'])

vaso_df_time_new = vaso_df_time.drop(columns = ['linkorderid', 'amount'])
vaso_df_time_new = vaso_df_time_new.rename(columns = {'vaso':'treatment'})

df_crystalloid_bolus = pd.read_csv("/Users/mayeshasoshi/Documents/Capstone/local_path/treatments/crystalloid_bolus.csv")
df_colloid_bolus = pd.read_csv("/Users/mayeshasoshi/Documents/Capstone/local_path/treatments/colloid_bolus.csv")

df_colloid_bolus = df_colloid_bolus.sort_values('charttime')

df_crystalloid_bolus = df_crystalloid_bolus.sort_values('charttime')

df_bolus = pd.DataFrame(columns=['stay_id', 'Time', 'Bolus_value'])

def add_to_df(df, time, id, bolus_value):
    new_entry = pd.DataFrame({'stay_id': [id], 'Time': [time], 'Bolus_value': [bolus_value]})
    df = pd.concat([df, new_entry], ignore_index=True)
    return df

colloid_i = 0
crystalloid_i = 0
while colloid_i < len(df_colloid_bolus) and crystalloid_i < len(df_crystalloid_bolus):
    colloid_bolus_stay_id = df_colloid_bolus.iloc[colloid_i]['stay_id']
    colloid_bolus_charttime = df_colloid_bolus.iloc[colloid_i]['charttime']
    colloid_bolus_colloid_bolus = df_colloid_bolus.iloc[colloid_i]['colloid_bolus']
    
    crystalloid_bolus_stay_id = df_crystalloid_bolus.iloc[colloid_i]['stay_id']
    crystalloid_bolus_charttime = df_crystalloid_bolus.iloc[colloid_i]['charttime']
    crystalloid_bolus_colloid_bolus = df_crystalloid_bolus.iloc[colloid_i]['crystalloid_bolus']
    
    timestamp_compare_val = compare_timestamps(colloid_bolus_charttime, crystalloid_bolus_charttime) == -1
    if timestamp_compare_val == -1:
        colloid_i +=1
        df_bolus = add_to_df(df_bolus, colloid_bolus_charttime, colloid_bolus_stay_id, colloid_bolus_colloid_bolus)
    elif timestamp_compare_val == 1:
        crystalloid_i +=1
        df_bolus = add_to_df(df_bolus, crystalloid_bolus_charttime, crystalloid_bolus_stay_id, crystalloid_bolus_colloid_bolus)
    elif timestamp_compare_val == 0:
        crystalloid_i+=1
        colloid_i +=1
        
        df_bolus = add_to_df(df_bolus, crystalloid_bolus_charttime, crystalloid_bolus_stay_id, str(int(colloid_bolus_colloid_bolus) + int(crystalloid_bolus_colloid_bolus)))

df_bolus['endtime'] = 'N/A'

# rename the chartime column to starttime 
df_bolus = df_bolus.rename(columns = {'Time':'starttime'})
df_bolus['treatment'] = 'fluid bolus'
df_bolus.set_index('stay_id', inplace=True)
df_treatments = pd.concat([df_bolus, vaso_df_time_new], axis=0)