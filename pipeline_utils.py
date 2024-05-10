import pandas as pd
from datapackage_io_util import *
import pickle
import warnings
import os
import re

# Load in static_data.csv and convert all features to the right datatype
# Argument:
#    data_path -- path of data directory
#    resource_path -- resource directory
# Return:
#   data_df -- dataframe of static info 
def load_static_df(data_path,resource_path):
    data_df = pd.read_csv(os.path.join(data_path,'static_data.csv'))
    schema = load_datapackage_schema(os.path.join(resource_path,'static_data_spec.json'))
    # Cast fields to required type (categorical / datetime)
    for ff, name in enumerate(schema.field_names):
        ff_spec = schema.descriptor['fields'][ff]
        if 'pandas_dtype' in ff_spec and ff_spec['pandas_dtype'] == 'category':
            data_df[name] = data_df[name].astype('category')
        elif 'type' in ff_spec and ff_spec['type'] == 'datetime':
            data_df[name] = pd.to_datetime(data_df[name])
    data_df = data_df.set_index('stay_id')
    return data_df

# regroup race; redefine the regrouping or skip if not desirable
# Argument:
#   df -- static dataframe
#   d -- regroup dictionary {original race: new race}
# Return:
#   df -- dataframe of static info with race regrouped
def regroup_ethnicity(df,d=None):
    if d is None: return df
    df['race_regroup'] = (df['race'].astype('object')).replace(d,regex=True)
    df['race_regroup'] = df['race_regroup'].str.lower()
    df['race_regroup'] = df['race_regroup'].astype('category')
    return df


# aggregate demographic info, comorbidity scores and sepsis label
# Argument:
#   data_path -- str: path of data folder
#   df -- dataframe: static info dataframe
#   demo_features -- list of strs: demographic features one want to select (default None and select all)
#   comorbidity -- list of strs: comorbidity scores one want to select (default None; options: columns in comorbidity/charlson.csv)
#   include_sepsis -- bool: if include sepsis-III label (default False)
# Return:
#   df -- dataframe aggregating demographic info, comorbidity scores and sepsis label
def aggr_static(data_path, \
                df, \
                demo_features = None, \
                comorbidity=None, \
                include_sepsis = False):
    # reset index
    df = df.reset_index()
    # select demographic features and features needed later
    static_df_fs = ['stay_id','subject_id','hadm_id','icu_intime','icu_outtime']
    if demo_features is not None:
        static_df_fs = demo_features+static_df_fs
    else:
        warnings.warn('Demographic feature not specified; None demographic feature selected') 
    df = df[static_df_fs]
    # concat comorbidity scores
    if (comorbidity is not None) and len(comorbidity)>0:
        fp = os.path.join(data_path,'comorbidity')
        print ('Reading charlson comorbidity score:',comorbidity)
        comorb_df = pd.read_csv(os.path.join(fp, 'charlson.csv'))
        # Concat static_df and commorbidity scores; only keep ids in static_df;    
        df = df.merge(comorb_df[comorbidity + ['hadm_id']],on='hadm_id',how='left')
    print ('Finished concat comorbidity scores... ')
    
    # concat sepsis labels
    if include_sepsis:
        fp = os.path.join(data_path,'sepsis')
        sepsis_df = pd.read_csv(os.path.join(fp, 'sepsis3.csv'))
        sepsis_df['sepsis3'] = 1.
        df = df.merge(sepsis_df[['sepsis3','stay_id']],on='stay_id',how='left')
        # if nan, then the label is 0
        df['sepsis3'] = df['sepsis3'].fillna(0)
    print ('Finished concat sepsis labels... ')
    df = df.set_index('stay_id')
    return df


def select_patient_cohort(df,
                          min_age = 18, max_age = 90,
                          min_duration = 12, max_duration = 240,
                          hosq_seq = None, icustay_seq = None,
                          mort_icu = None, mort_hosp = None,
                          readmission_30 = None):
    """
    Creates a patient cohort based on the selected criteria.

    Criteria:
    df (pandas.DataFrame): Static dataframe to be filtered.
    min_age (int): Minimum patient age used; defaults to 18.
    max_age (int): Maximum patient age used; defaults to 90.
    min_duration (int): Minimum length of stay in the ICU in hours; defaults to 12 hours.
    max_duration (int): Maximum length of stay in the ICU in hours; defaults to 240 hours.
    hosq_seq (int): Selects all hospital stays <= the value indicated; defaults to None.
    icustay_seq (int): Selects all ICU stays <= the value indicated; defaults to None.
    mort_hosp (Bool): Binary value limiting to patients that:
                      survived through hospital discharge (0),
                      or died during the hospital stay (1); defaults to None.
    mort_icu (Bool): Binary value limiting to patients that:
                     survived through ICU discharge (0),
                     or died in the ICU (1); defaults to None.
    readmission_30 (Bool): Binary value limiting to patients that:
                           were readmitted to the ICU in <= 30 days (1),
                           weren't readmitted within 30 days; defaults to None.
                    

    Returns:
    (pandas.DataFrame): Dataframe with the selected patient criteria.
    """
    # Filter patients by age 
    # (Set maximum_age to 1 if early newborns are wanted)
    if min_age >= max_age:
        raise AssertionError("Invalid age range parameters.")
    
    # Limit to needed age range
    if max_age is not None:
        df = df.loc[(np.ceil(df.age) >= min_age) & (np.ceil(df.age) <= max_age)]
    else:
        df = df.loc[np.ceil(df.age) >= min_age]
    
    # Limit to stays within a certain time frame (in hours)
    if min_duration is not None:
        df = df.loc[df.los_icu * 24 >= min_duration]
    if max_duration is not None:
        df = df.loc[df.los_icu * 24 <= max_duration]
    
    # Limit to hosq_seq numbered hospital stays and greater
    if hosq_seq is not None:
        df = df.loc[df.hosq_seq <= hosq_seq]
    
    # Limit to icustay_seq numbered ICU stays and greater
    if icustay_seq is not None:
        df = df.loc[df.icustay_seq <= icustay_seq]
    
    # Filter hospital stays according mortality status
    if mort_hosp is not None:
        df = df.loc[df.mort_hosp == mort_hosp]
    # Filter ICU stays according mortality status
    if mort_icu is not None:
        df = df.loc[df.mort_icu == mort_icu]
    
    # Filter stays according to readmission time frame
    if readmission_30 is not None:
        df = df.loc[df.readmission_30 == readmission_30]
    
    return df


# Read variable range table (Note: the table may not include all variables) 
# Argument: 
#   resource_path -- path of resource folder
# Return:
#   var_range -- table of variable ranges
def get_variable_ranges(resource_path):
    # Read in the second level mapping of the itemid, and take those values out
    columns = [ 'LEVEL2', 'OUTLIER LOW', 'VALID LOW', 'IMPUTE', 'VALID HIGH', 'OUTLIER HIGH' ]
    to_rename = dict(zip(columns, [ c.replace(' ', '_') for c in columns ]))
    to_rename['LEVEL2'] = 'VARIABLE'
    var_ranges = pd.read_csv(os.path.join(resource_path,'variable_ranges.csv'), index_col=None)
    var_ranges = var_ranges[columns]
    var_ranges.rename(columns=to_rename, inplace=True)
    var_ranges = var_ranges.drop_duplicates(subset='VARIABLE', keep='first')
    var_ranges['VARIABLE'] = var_ranges['VARIABLE'].str.lower().str.replace(' ','_')
    var_ranges.set_index('VARIABLE', inplace=True)
    var_ranges = var_ranges.loc[var_ranges.notnull().all(axis=1)]

    return var_ranges

def fahrenheit_to_celsius(f):
    return (5/9) * (f - 32)

def aggr_dynamic_features(selected_features_and_levels=[],
                          icu_ids=[],
                          data_path='',
                          variable_ranges=None,
                          exclusion_method='None',
                          alternative_method='None',
                          low_quantile = 0.01, 
                          high_quantile = 0.99,
                          median_abs_devs = 5):
    """
    Preprocesses and selects the dynamic features needed for analysis.

    Criteria:
        selected_features_and_levels (list): 
            List of features and levels to be selected.
            Format: [feature_name: str | [feature_name: str, level1: str] ...]
        icu_ids (list): 
            List of ICU stay IDs to be kept after preprocessing.
        data_path (str): 
            Path to directory containing reading files.
        use_table (Bool): 
            Whether to use the outlier/extreme value tables available for some features.
            If set to False we default to the alternative method.
        variable_ranges (pandas.DataFrame): 
            Outlier/extreme value ranges for some features. Format:
                                OUTLIER_LOW	VALID_LOW	IMPUTE	VALID_HIGH	OUTLIER_HIGH
                VARIABLE					
                alanine_aminotransferase	0.0	2.00	34.00	10000.00	11000.0
                albumin	                    0.0	0.60	3.10	6.00	60.0
                ...
        exclusion_method (str): 
            Whether to use the valid high/low points or outlier high/low points from
            the outlier/extreme value table. One of: 
                'Valid'
                'Outlier'
        alternative_method (str): 
            What method to use to remove outliers from features. One of:
                'None': Default method; outliers are kept.
                'Z-Score': Use quantiles to set cutoff limits.
                'MAD': Use Median Absolute Deviation method to set cutoff limits.
        low_quantile (float): 
            Lower quantile cutoff if using the z-score method.
        high_quantile (float): 
            Higher quantile cutoff if using the z-score method.
        median_abs_devs (int): 
            Number of median abosolute deviations from the median to use as the 
            outlier cutoff if using the MAD method.

    Returns:
        union_df (pandas.DataFrame): All of the preprocessed vitals/labs indexed by ICU stay ID & event time.
    """

    feature_path = os.path.join(data_path, 'vitals_and_labs')
    union_df = None
            
    # Iterate over selected features
    for idx, feature in enumerate(selected_features_and_levels):
        levels = feature.split('(')
        feature = levels[0]
        # print("os.path.join(feature_path, feature + '.csv'): ", os.path.join(feature_path, feature + '.csv'))
        feature_df = pd.read_csv(os.path.join(feature_path, feature + '.csv'))
        # If feature is GCS or urine output, rename column to 'valuenum'
        # These are the only vitals/labs with this discrepancy
        if feature in ['gcs', 'urine_output']:
            feature_df = feature_df.rename(columns = {''.join(feature.split('_')):"valuenum"})
            
        print('Finished reading {}.csv'.format(feature))
        
        # Filter data with ICU stay IDs selected with static information
        feature_df = feature_df.loc[feature_df.stay_id.isin(icu_ids)]
        
        # Throw a warning if a selected dataframe is empty
        if len(feature_df) == 0:
            warnings.warn(f'The {feature} feature has no relevant entries for the selected cohort.\nIt will be excluded from here on out.')
            continue
        
        # The value 999,999 is a special NA-type code
        feature_df = feature_df[feature_df.valuenum != 999999]
        
        # Filter data if level 1 specified
        if len(levels) > 1:  
            level1 = '{} ({})'.format(feature.replace('_',' '), levels[1][:-1])
            feature_df = feature_df.loc[feature_df.level1 == level1]
            
        # Convert all temperatures to Celsius
        if feature == 'temperature':
            
            feature_df['valuenum'] = feature_df.apply(
                lambda row: fahrenheit_to_celsius(row['valuenum']) if row['level1'] == 'temperature (f)' else row['valuenum'],
                axis = 1)

        # Convert weights to kilograms
        if feature == 'weight':
            feature_df.loc[feature_df.level1 == 'weight (lbs, admission)', 'valuenum'] *= 0.45359237
        
        # Exclude points based on outlier/extreme value table
        if variable_ranges is not None and feature in variable_ranges.index:
            
            if exclusion_method == 'Outlier':
                feature_df = feature_df.loc[(feature_df.valuenum >= variable_ranges.loc[feature]['OUTLIER_LOW'])&\
                                            (feature_df.valuenum <= variable_ranges.loc[feature]['OUTLIER_HIGH'])]
            elif exclusion_method == 'Valid':
                feature_df = feature_df.loc[(feature_df.valuenum >= variable_ranges.loc[feature]['VALID_LOW'])&\
                                            (feature_df.valuenum <= variable_ranges.loc[feature]['VALID_HIGH'])]
            else:
                raise Exception('Invalid exclusion option.')
        
        # Use z-score to find and exclude outliers
        elif alternative_method == 'Z-Score':
            
            assert 0 <= low_quantile <= 1.0, 'Use valid lower quantile bound.'
            assert 0 <= high_quantile <= 1.0, 'Use valid higher quantile bound.'
            assert low_quantile <= high_quantile, 'Use valid lower & higher quantile bounds.'
            
            feature_df = feature_df.loc[(feature_df.valuenum >= feature_df.valuenum.quantile(low_quantile))&\
                                        (feature_df.valuenum <= feature_df.valuenum.quantile(high_quantile))]
        
        # Use median absolute deviations to exclude outliers
        elif alternative_method == 'MAD':
            
            median = feature_df.median()
            mad = (feature_df - median).abs().median()
            lower_bound = median - median_abs_devs * mad
            upper_bound = median + median_abs_devs * mad
            feature_df = feature_df.loc[(feature_df >= lower_bound) & (feature_df <= upper_bound)]  
        
        # Throw an error for invalid outlier methods
        elif alternative_method != 'None':
            raise Exception('Invalid outlier method.')
            
        # Concatenate all features into one big table
        feature_df = feature_df[['stay_id','charttime','valuenum']]
        feature_df.rename(columns = {'valuenum': feature}, inplace = True)
        
        if union_df is None:
            union_df = feature_df.copy()
        else:
            union_df = union_df.merge(feature_df, how = 'outer')
        
    # Sort by stay ID and event time
    union_df = union_df.set_index(['stay_id', 'charttime']).sort_index()
    
    return union_df

# # ================Unit conversion============
#         # (fio2/spo2/sao2/hematocrit: numeric --> %)
#         if f in ['oxygen_saturation','hematocrit']:
#             fdf['valuenum'] = fdf.valuenum.apply(lambda x: x*100. if x<=1. else x)
#         # (fraction_inspired_oxygen: between 0 & 1)
#         if f == 'fraction_inspired_oxygen':
#             fdf['valuenum'] = fdf.valuenum.apply(lambda x: x/100. if x>1. else x)
#         # if larger than 100, divide by 100
#         if f == 'calcium_ionized':
#             fdf['valuenum'] = fdf.valuenum.apply(lambda x: x/100. if x>90. else x)

# Help function that aggregate treatments with dynamic features and return first treatment timestamp
# Arguments
#   tr_df -- dictionary with treatments as keys and treatment dataframes as values
#   df -- dataframe of patients vitals and labs
#   icuid -- stay_id of the patient
#   binning -- how to bin treatment (irregular versus regular)

# return
#   df -- aggregated treatment dataframe
def aggr_treatments(tr_df, df, icuid, binning):
    for tr,fdf in tr_df.items():
        bins = df.index
        bins = [b.left for b in bins]+[bins[-1].right]

        # initialize treatment value with 0
        df[tr+'(binary)'] = np.zeros(len(df))
        # if vaso of fluid, initialize amount column
        if tr not in ['ventilation','invasive_line','antibiotic']:
            df[tr+'(amount)'] = np.zeros(len(df))
        # If no treatment given, continue
        if icuid not in fdf.index: 
            continue
        fdf = fdf.loc[[icuid]]

        # drop treatment with starttime after last timestamp (could happen if the traj is cut....)
        fdf = fdf[fdf.starttime < df.index[-1].right]
        if len(fdf) == 0: 
            continue

        if tr not in ['ventilation','invasive_line','antibiotic']:
            amount = pd.Series(fdf['amount']).values

        # determine start time interval bin;
        st = pd.Series(fdf.loc[icuid,'starttime']).values
        st_bins = pd.cut(st,bins=bins,right=False)
        # assert all starting timestamps are within the range
        assert (~st_bins.isna()).all()
        
        if tr[-5:]=='bolus':
            df.loc[st_bins,tr+'(amount)']=amount
            df.loc[st_bins,tr+'(binary)']=np.ones(len(st))
        else:
            # determine end time interval bin; if not in range, use last timestamp
            et = (pd.Series(fdf['endtime']) ).values
            et_bins = pd.cut(et- pd.Timedelta(seconds=1),bins=bins,right=False)
            et_bins = et_bins.fillna(df.index[-1])

            st = pd.to_datetime(st)
            et = pd.to_datetime(et)
            assert len(st) == len(et)
            for i in range(len(st)):
                df.loc[st_bins[i]:et_bins[i],tr+'(binary)'] = 1.
                if tr not in ['ventilation','invasive_line','antibiotic']:
                    l = df.loc[st_bins[i]:et_bins[i]].shape[0]
                    p = np.ones(l)
                    
                    if binning=='irregular' and l!=1:
                        p[0] = (st_bins[i].right-st[i]).total_seconds()/3600
                        p[-1] = (et[i]-et_bins[i].left).total_seconds()/3600
                    assert (p>=0.).all()
                    p = p/np.sum(p)
                    df.loc[st_bins[i]:et_bins[i],tr+'(amount)'] += (p*amount[i])
    return df

# Help function to normalize vasopressor dosage; 
# If vaso type not implemented, return 0 (not added to the normalized vaso amount)
# Arguments
#   vaso_name -- name of the vasopressor
#   amount -- original dosage
# Return    converted dosage
def normalizing_vaso(vaso_name,amount):
    if vaso_name == 'vasopressin': return amount*2.5
    elif vaso_name == 'phenylephrine': return amount*0.1
    elif vaso_name == 'dopamine': return amount*0.01
    elif vaso_name in ['epinephrine','norepinephrine']: return amount
    else: 
        warnings.warn('Normalization not implemented for '+vaso_name + '; excluded from normalized vaso!')
        return amount*0.

# Help function to load treatments and exclude extreme values
# Arguments
#   data_path -- path of data folder
#   icuids -- list of icuids to be included
#   treatments -- list of treatments names
# Return
# tr_df -- dictionary with treatment names as keys and treatment dataframes as values
def load_treatments(data_path, icuids, treatments):
    # no imputation for treatment
    tr_d = {}
    if treatments is None: return tr_d
    # Load in treatments data 
    for tr in treatments:
        fp = os.path.join(data_path,'treatments')
        fdf = pd.read_csv(os.path.join(fp,tr+'.csv'))
        print ('finish reading {}.csv'.format(tr))

        if tr[-5:] == 'bolus':
            fdf['starttime'] = fdf['charttime']
            fdf = fdf.rename(columns={tr:'amount'})
            # filter out colus larger than 10L
            fdf = fdf.loc[fdf.amount < 10000]
            fdf.drop(columns=['charttime'],inplace=True)
        elif tr in ['invasive_line','ventilation', 'antibiotic']:
            if tr == 'antibiotic':
                fdf = fdf.rename(columns={'stoptime':'endtime'})
            fdf['endtime'] = pd.to_datetime(fdf['endtime'])
        else:
            if tr == 'neuroblock':
                fdf = fdf.rename(columns={'drug_amount':'amount'})
            else:    
                fdf = fdf.rename(columns={'vaso_amount':'amount'})
            fdf['endtime'] = pd.to_datetime(fdf['endtime'])

        fdf['starttime'] = pd.to_datetime(fdf['starttime'])
        fdf = fdf.loc[fdf.stay_id.isin(icuids)].set_index('stay_id')

        # Only consider Invasive ventilation (change if needed)
        if tr=='ventilation': fdf = fdf.loc[fdf.ventilation_status=='InvasiveVent']
        tr_d[tr] = fdf
    return tr_d




# Aggregate vitals and labs with static features (if any) and treatments (if any)
# Arguments
#   dyn_df -- big union table of all vitals and labs from previous step
#   impute_tb -- Table of imputation values
#   static_df -- static info dataframe (default none)
#   tr_df -- treatment dataframe dictionary from previous step (default {})
#   traj_cut -- if cut the trajectory to first N hours (default 100000 (keep all))
#   t_w -- feature binning time window length in hour (default 1)
#   start_ts -- starting timestamp (if admit, using icu admission time; if treatment, use timestamp of first treatment; default treatment)
#   binning -- how to bin treatment (irregular versus regular; default: irregular)
#   norm_treatment -- if normalizing and aggregate treatment dosage, including vaso and fluid 
#                           (default true; note not all vasos are implemented)
#   fill_na -- if filling nans (if true use forward filling; if there are still nans, use impute table )
def feature_aggr(dyn_df, \
                 impute_tb,\
                 icuids, \
                 tr_df = {},\
                 static_df = None, \
                 t_w = 1., \
                 traj_cut = 100000, \
                 start_ts = 'treatment', \
                 binning='irregular',\
                 norm_treatment = True, \
                 fill_na = True):
    # couple sanity checks
    if len(tr_df)==0 and start_ts=='treatment': 
        raise Exception ('Use the first treatment as the starting timestamp. Please specify treatments')
    if static_df is None or len(static_df)==0: 
        raise Exception ('Please pass static dataframe')
    if start_ts not in ['admit','treatment']:    
        raise Exception('Invalid starting timestamp option!')
    
    # Start aggregation
    patient_data_list = list()
    for i,icuid in enumerate(icuids):
        static_info = static_df.loc[icuid]
        
        # Determine start time of dataframe (icu admission time or time of first treatment given)
        tr_ts = [pd.Series(v.loc[icuid,'starttime']).min() for k,v in tr_df.items() if icuid in v.index]
        if start_ts == 'admit':
            tr_ts = [static_info.icu_intime]+tr_ts
        start = min(tr_ts)
        
        n_period = int(np.ceil((static_info.icu_outtime - static_info.icu_intime).total_seconds()/3600.))//t_w
        n_period = int(min(n_period, traj_cut))
        
        # sort vitals & labs by time
        u = dyn_df.loc[icuid].sort_values('charttime')
        # aggregate values to the specified time window
        u['time']=pd.to_datetime(u.index)

        bins = pd.date_range(start, periods=n_period+1, freq='{}min'.format(int(t_w*60)))
        u = u.groupby([pd.cut(u['time'], bins, right=False)]).mean()
        # aggregate treatments and dynamic features
        u = aggr_treatments(tr_df,u,icuid,binning)
        # impute (carry forward; then fill missing value with impute table)
        if fill_na:
            u = u.ffill()
            u = u.fillna(impute_tb)
        u = pd.concat({icuid:u},names=['stay_id']).reset_index()
        patient_data_list.append(u)
        if (i+1)%500==0: print ('Finish process # {} patients'.format(int(i)))
    # concat data of all patients
    aggr_df = pd.concat(patient_data_list,ignore_index=True).set_index(['stay_id','time'])
    # normalize treatment
    if norm_treatment:
        aggr_df['vaso(binary)']= np.zeros(len(aggr_df))
        aggr_df['vaso(amount)']= np.zeros(len(aggr_df))
        aggr_df['bolus(binary)']= np.zeros(len(aggr_df))
        aggr_df['bolus(amount)']= np.zeros(len(aggr_df))
        for tr in tr_df.keys():
            print ('normalizing {} ...'.format(tr))
            if tr=='ventilation':continue
            elif  tr[-5:]=='bolus':
                aggr_df['bolus(amount)'] += (aggr_df[tr+'(amount)'])
                aggr_df['bolus(binary)'] = aggr_df[[tr+'(binary)','bolus(binary)']].max(axis=1)
            else:
                aggr_df['vaso(amount)'] += (normalizing_vaso(tr,aggr_df[tr+'(amount)']))
                aggr_df['vaso(binary)'] = aggr_df[[tr+'(binary)','vaso(binary)']].max(axis=1)
    # merge with static info if any
    if static_df is None: 
        warning.warn('Static dataframe not passed.')
    else:
        # join with static features (demographic + score)
        aggr_df = aggr_df.join(static_df,how='inner')
    aggr_df = aggr_df.drop(columns =['icu_intime', 'icu_outtime','subject_id','hadm_id'])
    return aggr_df





# a function that takes in a data_path, and a text file that contains the list 
# of features that I want to aggregate, and how I want to aggregate them and 
# then it will return the union_df and impute_tb
"""
the JSON file should be a list: 
    The first element should be the path of the static dataframe
    The second element should be the path of the variable ranges
    Then, the operations are the objects that follow in the list,
    and they are ran in the order they are in the list

script.json:
[
    {"static_dataframe_path": "str (path of static_data.csv)"},
    {"variable_ranges_path": "str (path of variable_ranges.csv)"},
    {
        "regroup_ethnicity": {
            "WHITE.*": "white",
            "ASIAN.*": "asian",
            "BLACK.*": "black",
            "AMERICAN INDIAN.*":"amer indian",
            "CARIBBEAN ISLAND":"other",
            "HISPANIC.*":"hisp/latino",
            "MULTI RACE ETHNICITY":"other",
            "PATIENT DECLINED TO ANSWER":"other",
            "PORTUGUESE":"south amer/port",
            "SOUTH AMERICAN":"south amer/port",
            "UNABLE TO OBTAIN":"other",
            "UNKNOWN.*":"other",
            ".*PACIFIC ISLANDER":"other",
            "OTHER":"other",
            "AMERICAN":"other"
        }
    },

    {
        "select_patient_cohort": {
            "min_age": "float | None (minimal age; default 18, youngest patient in cohort is 18)",
            "max_age": "float | None (maximal age; default 90, ages of patients older than 90 may have been shifted)",
            "min_duration": "float | None (minimal length of stay in icu in hours; default 12 hrs)",
            "max_duration": "float | None (maximal length of stay in icu in hours; default 240 hrs, varies from hours to months)",
            "hosq_seq": "int | None (<= i-th time of vising hospital; ranges from 1-30, default none and select all)",
            "icustay_seq": "int | None (<= i-th time of admission icu; ranges from 1-7, default none and select all)",
            "mort_hosp": "int | None (hospital mortality; 0-survival to hospital discharge, 1-death in hosp, default none and select all)",
            "mort_icu": "int | None (icu mortality; 0-survival to icu discharge, 1-death in icu, default none and select all)",
            "readmission_30": "int | None (readmission in 30 days; 0-no readmission, 1-readmission to icu, default none and select all)"
        }
    },


    {
        "dynamic_features": [
            {
                "selected_features_and_levels": [feature_name: str | [feature_name: str, level1: str] ...]
                "exclusion_method":  "Valid" | "Outlier" (default "None"),
                "imputation_method": "Mean" | "Median" | "Mode" (default "Mean"),
                "low_quantile": float (default 0),
                "high_quantile": float (default 1),
            },
            {
                "selected_features_and_levels": [feature_name: str | [feature_name: str, level1: str] ...]
                "exclusion_method":  "Valid" | "Outlier" (default "None"),
                ...
            },
            ...
        ]
    },

    "treatments": [
        "treatment: str (treatment name)",
        "treatment: str (treatment name)",
        ...
    ]
]
"""
def compare_timestamps(time1, time2):
    # Define the date format
    format = "%Y-%m-%d %H:%M:%S"
    
    # Convert string to datetime objects
    datetime1 = datetime.strptime(time1, format)
    datetime2 = datetime.strptime(time2, format)
    
    # Compare the two datetime objects
    if datetime1 < datetime2:
        return -1
    elif datetime1 > datetime2:
        return 1
    else:
        return 0

def add_to_df(df, time, id, bolus_value):
    new_entry = pd.DataFrame({'stay_id': [id], 'starttime': [time], 'Bolus_value': [bolus_value]})
    return pd.concat([df, new_entry], ignore_index=True)

def merge_bolus_data(crystalloid, colloid):
    df_bolus = pd.DataFrame(columns=['stay_id', 'starttime', 'Bolus_value'])
    colloid_i = 0
    crystalloid_i = 0

    while colloid_i < len(colloid) and crystalloid_i < len(crystalloid):
        colloid_entry = colloid.iloc[colloid_i]
        crystalloid_entry = crystalloid.iloc[crystalloid_i]
        timestamp_compare_val = compare_timestamps(colloid_entry['charttime'], crystalloid_entry['charttime'])

        if timestamp_compare_val == -1:
            colloid_i += 1
            df_bolus = add_to_df(df_bolus, colloid_entry['charttime'], colloid_entry['stay_id'], colloid_entry['colloid_bolus'])
        elif timestamp_compare_val == 1:
            crystalloid_i += 1
            df_bolus = add_to_df(df_bolus, crystalloid_entry['charttime'], crystalloid_entry['stay_id'], crystalloid_entry['crystalloid_bolus'])
        else:
            colloid_i += 1
            crystalloid_i += 1
            total_bolus = str(int(colloid_entry['colloid_bolus']) + int(crystalloid_entry['crystalloid_bolus']))
            df_bolus = add_to_df(df_bolus, crystalloid_entry['charttime'], crystalloid_entry['stay_id'], total_bolus)

    df_bolus['endtime'] = 'N/A'
    df_bolus.rename(columns={'Time': 'starttime'}, inplace=True)

    return df_bolus

def process_treatments(labs_df, data_path, treatments, crystalloid_path, colloid_path):
    # Load unique stay IDs and treatment data
    icuids = labs_df['stay_id'].unique()
    vaso_treatments = load_treatments(data_path, icuids, treatments)

    # Process vasopressor treatments
    for drug, multiplier in [('vasopressin', 5), ('phenylephrine', 0.45), ('dopamine', 0.01)]:
        vaso_treatments[drug]['vaso_rate'] *= multiplier
        if drug == 'vasopressin':
            vaso_treatments[drug] = vaso_treatments[drug][vaso_treatments[drug]['vaso_rate'] <= 0.2]

    # Combine treatments into a single DataFrame
    vaso_df = pd.concat([v.assign(treatment=drug) for drug, v in vaso_treatments.items()], axis=0)
    vaso_df = vaso_df.sort_values(by=['stay_id', 'starttime']).drop(columns=['linkorderid', 'amount'])
    
    # Load bolus data and sort
    df_crystalloid_bolus = pd.read_csv(crystalloid_path).sort_values('charttime')
    df_colloid_bolus = pd.read_csv(colloid_path).sort_values('charttime')

    # Merge bolus data
    df_bolus = merge_bolus_data(df_crystalloid_bolus, df_colloid_bolus)

    # Combine all treatments
    df_bolus['treatment'] = 'fluid bolus'
    df_bolus.set_index('stay_id', inplace=True)
    df_treatments = pd.concat([df_bolus, vaso_df], axis=0)

    return df_treatments



def json_to_cohort_df(file_path):
    import json
    # load in the json file and make sure it is a list
    with open(file_path, 'r') as f:
        json_dict = json.load(f)
    assert type(json_dict) == list, 'JSON file must be a list'

    # ----------------------------------------------------------------------------------
    # ---- asserting errors and loading in the static dataframe and variable ranges ----
    # ----------------------------------------------------------------------------------

    # assert that the first two elements are the paths to the static dataframe and variable ranges
    assert 'static_dataframe_path' in json_dict[0].keys(), 'static_dataframe_path must be provided as the first element in the list'
    assert 'variable_ranges_path' in json_dict[1].keys(), 'variable_ranges_path must be provided as the second element in the list'
    # read in the paths
    static_dataframe_path = json_dict[0]['static_dataframe_path']
    variable_ranges_path = json_dict[1]['variable_ranges_path']
    resource_path = json_dict[2]['resource_path']
    # load in the static dataframe and variable ranges
    print(f'Loading static dataframe from {static_dataframe_path}...')
    static_df = load_static_df(static_dataframe_path, resource_path)
    print(f'Loading variable ranges from {variable_ranges_path}...')
    variable_ranges_df = get_variable_ranges(variable_ranges_path)

    # ----------------------------------------------------------------------------------
    # ---------------------- run operations in the json file ---------------------------
    # ----------------------------------------------------------------------------------
    aggr_df = None

    for operation in json_dict[3:]:
        # if operation is select_patient_cohort
        print("operation: ", operation)
        if 'select_patient_cohort' in operation.keys():
            # get the function arguments
            args_select_patient_cohort = operation['select_patient_cohort']
            # assert that the arguments are in the correct format
            assert type(args_select_patient_cohort) == dict, 'select_patient_cohort must be a dictionary'
            # run the function
            selected_static_df = select_patient_cohort(static_df, **args_select_patient_cohort)
        # if operation is dynamic_features
            selected_patients = selected_static_df[selected_static_df['icustay_seq'] != 1]['subject_id'].unique()
            selected_static_df_new = selected_static_df[selected_static_df['subject_id'].isin(selected_patients)]
            selected_static_df_new = selected_static_df_new[selected_static_df_new['icustay_seq'] == 1]
        if 'dynamic_features' in operation.keys():
            # get the features
            list_dynamic_features = operation['dynamic_features']
            # assert that the features are in the correct format
            assert type(list_dynamic_features) == list, 'dynamic_features must be a list'
            # loop through the features
            for feature in list_dynamic_features:
                # assert that the feature is a dictionary
                assert type(feature) == dict, 'every feature in dynamic_features must be a dictionary'
                # run the function
                print(f'Aggregating feature: {feature["feature"]}...')
                # run the function
                aggr_df = aggr_dynamic_features(selected_features_and_levels=feature['feature'], 
                                                variable_ranges=variable_ranges_df, 
                                                data_path=feature['data_path'], 
                                                icu_ids=selected_static_df_new.index.to_list(), 
                                                exclusion_method=feature["exclusion_method"])
        
        if 'treatments' in operation.keys():
            
            treatment_cohort = operation['treatments']
            
            treatment_df = process_treatments(labs_df=aggr_df, 
                               data_path=treatment_cohort['data_path'], 
                               treatments=treatment_cohort['treatments'], 
                               crystalloid_path=treatment_cohort['crystalloid_path'],
                               colloid_path=treatment_cohort['colloid_path'],
                               )
            
            # # get the features
            # list_dynamic_features = operation['dynamic_features']
            # # assert that the features are in the correct format
            # assert type(list_dynamic_features) == list, 'dynamic_features must be a list'
            # # loop through the features
            # for feature in list_dynamic_features:
            #     # assert that the feature is a dictionary
            #     assert type(feature) == dict, 'every feature in dynamic_features must be a dictionary'
            #     # run the function
            #     print(f'Aggregating feature: {feature["feature"]}...')
            #     # run the function
            #     aggr_df = aggr_dynamic_features(selected_features_and_levels=feature['feature'], 
            #                                     variable_ranges=variable_ranges_df, 
            #                                     data_path=feature['data_path'], 
            #                                     icu_ids=selected_static_df.index.to_list(), 
            #                                     exclusion_method=feature["exclusion_method"])
                
        else:
            raise Exception('Invalid operation in JSON file')
        
    return aggr_df, selected_static_df, variable_ranges_df, treatment_df

