import argparse
import pathlib
import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import table
from textwrap import wrap

audiogram_colors = {
        "normal": "BLUE",
        "mild": "GREEN",
        "moderate": "ORANGE",
        "severe": "RED"
    }

def sigmoid(x):
    # Range between -1 and 1
    # Clip and scale x to prevent occasional RuntimeWarning: overflow encountered in exp
    x = np.clip(x, -512, 512)
    
    # Adjust scale based on the magnitude of x
    scale = 1.0
    if np.abs(x).max() > 100:
        scale = 0.01
    elif np.abs(x).max() > 10:
        scale = 0.1

    x = x * scale
    return 2 / (1 + np.exp(-x)) - 1


def get_score(ref, proc):
    """Get score between two evaluation metrics

    Args:
        ref: original evaluation metric
        proc: processed evaluation metric

    Returns:
        average: average score between two metrics
    """
    # Check lengths
    # if len(ref) != len(proc):
    #     # Just diff the averages
    #     diff = np.mean(proc, dtype=np.float64) - np.mean(ref, dtype=np.float64)
    # else:
    # Compute a difference between each frame of the two metrics
    diff = proc - ref

    # Normalize between -1 and 1
    score = sigmoid(diff)

    # Average
    average = np.mean(score, dtype=np.float64)
    
    return average

def score_data(data: pd.DataFrame) -> pd.DataFrame:
    """Score the data for spectral features
    
    Args:
        data: The data from the analysis results file
        
    Returns:
        scored_data: The scored data for evaluation metrics
    """
    score_data = data.copy()

    for feature in ['voiced_probabilities', 'spectral_flatness', 'harmonic_energy']:
        # Group by 'sample_id'
        for sample_id, group in score_data.groupby('sample_id'):
            # Extract the reference value for the sample_id
            voice_prob_ref = group.loc[group['f0_ratio'] == 'ref', feature].values[0]
            
            # Get score for each f0_mix value compared to the ref value
            for idx, row in group.iterrows():
                # Update the feature column with a sigmoid score
                score_data.at[idx, feature] = get_score(voice_prob_ref, row[feature])
        
        # Finally, update the name of the feature column
        score_data.rename(columns={feature: f'{feature}_score'}, inplace=True)
     
    return score_data

def normalize_for_spectral(data: dict) -> pd.DataFrame:
    """Flatten the data for spectral features analysis
    
    Args:
        data: The data from the analysis results file
        
    Returns:
        norm_data: The normalized data for spectral features analysis
    """
    norm_data = []

    for sample_id, sample_details in data.items():
        if 'spectral' in sample_details:
            spectral_data = sample_details.get('spectral', {})
        else:
            raise KeyError('The key "spectral" does not exist in sample_details; try running samplifi again with --eval-spectral')
        spectral_data = sample_details.get('spectral', {})
        for audiogram_profile, audiogram_data in spectral_data.items():
            for f0_ratio, f0_mix_data in audiogram_data.items():
                instrument = f0_mix_data.get('instrument', 'unknown')
                v_prob = np.nan_to_num(f0_mix_data.get('voiced_probabilities', None))
                s_flat = np.nan_to_num(f0_mix_data.get('spectral_flatness', None))
                h_energy = np.nan_to_num(f0_mix_data.get('harmonic_energy', None))
                mfcc_sim = np.nan_to_num(f0_mix_data.get('mfcc_similarity', None))
                norm_data.append({
                    'sample_id': sample_id,
                    'audiogram_profile': audiogram_profile,
                    'f0_ratio': f0_ratio,
                    'instrument': instrument,
                    'voiced_probabilities': v_prob,
                    'spectral_flatness': s_flat,
                    'harmonic_energy': h_energy,
                    'mfcc_similarity': mfcc_sim
                })

    # Convert to DataFrame
    return pd.DataFrame(norm_data)

def normalize_for_haaqi(data: dict) -> pd.DataFrame:
    """Flatten the data for HAAQI analysis
    
    Args:
        data: The data from the analysis results file
        
    Returns:
        norm_data: The normalized data for HAAQI analysis
    """
    norm_data = []

    for sample_id, sample_details in data.items():
        if 'haaqi' in sample_details:
            haaqi_data = sample_details.get('haaqi', {})
        else:
            raise KeyError('The key "haaqi" does not exist in sample_details; try running samplifi again with --eval-haaqi')
        for audiogram_profile, audiogram_data in haaqi_data.items():
            for f0_ratio, f0_mix_data in audiogram_data.items():
                instrument = f0_mix_data.get('instrument', 'unknown')
                # Not expecting NaNs from HAAQI
                haaqi_score = f0_mix_data.get('score', None)
                norm_data.append({
                    'sample_id': sample_id,
                    'audiogram_profile': audiogram_profile,
                    'f0_ratio': f0_ratio,
                    'instrument': instrument,
                    'haaqi_scores': haaqi_score,
                })

    # Convert to DataFrame
    return pd.DataFrame(norm_data)

def create_table_for_feature(data: pd.DataFrame, feature: str, folder: pathlib.Path) -> None:
    """Create the results table for the given feature

    Args:
        sample: The sample dataframe

    Returns:
        table: The table for the given feature
    """
    # Check for invalid feature
    if feature not in data.columns:
        raise ValueError(f'The feature "{feature}" does not exist in the DataFrame.')
    
    rows = ['normal', 'mild', 'moderate', 'severe']
    columns = ['ref', '0.25_mix', '0.5_mix', '0.75_mix', 'f0']
    if feature == 'haaqi_scores':
        columns = ['0.25_mix', '0.5_mix', '0.75_mix', 'f0']
    
    # Create a pivot table
    pivot_table = data.pivot_table(index='audiogram_profile', 
                                   columns='f0_ratio', 
                                   values=feature, 
                                   aggfunc='mean')
    
    # Round the values so it doesn't look messy
    pivot_table = pivot_table.round(3)

    # Reindex the table
    if rows:
        pivot_table = pivot_table.reindex(rows, axis=0)
    pivot_table = pivot_table.reindex(columns, axis=1)

    
    _, ax = plt.subplots(figsize=(10, 2))  # Adjust figsize to fit your data
    ax.axis('off')  # Hide axes

    # Create and style the table
    tbl = table(ax, pivot_table, loc='center', cellLoc='center', rowLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1.4, 1.4)  # Adjust scale to fit your data
    tbl.auto_set_column_width(col=list(range(len(columns))))  # Adjust column widths

    # Style header cells
    for key, cell in tbl.get_celld().items():
        if key[0] == 0 or key[1] == -1:  # Header row or index column
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#CCCCCC')  # Light grey
        else:
            cell.set_facecolor('#F4F4F4')  # Lighter grey for data cells

    plt.suptitle(f'Average {feature} across dataset', fontsize=16, y=0.95)
    
    plt.savefig(folder.joinpath(f'average_{feature}_table.png'), bbox_inches='tight')
    plt.close()

    return pivot_table


def create_graph_for_feature(data: pd.DataFrame, feature: str, folder: pathlib.Path) -> None:
    """Create the results graph for the given feature

    Args:
        sample: The sample dataframe

    Returns:
        graph: The graph for the given feature
    """
    # Check for invalid feature
    if feature not in data.columns:
        raise ValueError(f'The feature "{feature}" does not exist in the DataFrame.')
    
    markers = ['o', 'x', 's', '^']
    x_axis_vals = ['ref', '0.25_mix', '0.5_mix', '0.75_mix', 'f0']

    # Change order of x-axis values
    data['f0_ratio'] = pd.Categorical(data['f0_ratio'], x_axis_vals, ordered=True)

    # Aggregate the data
    aggregate_data = data.groupby(['audiogram_profile', 'f0_ratio'], observed=True)[feature].mean().reset_index()

    # Plot a line for each audiogram_profile
    for profile, color, marker in zip(audiogram_colors.keys(), audiogram_colors.values(), markers):
        # Filter the data for the current profile
        profile_data = aggregate_data[aggregate_data['audiogram_profile'] == profile]
        
        # Sort the values by f0_ratio to ensure the line is plotted correctly
        profile_data = profile_data.sort_values(by='f0_ratio')
        
        # Plot
        plt.plot(profile_data['f0_ratio'], profile_data[feature], marker=marker, color=color, label=profile)

    # Adding graph elements
    plt.title(f'{feature.capitalize()} across F0 Mixture Ratios')
    plt.xlabel('F0 Mixture Ratio')
    plt.ylabel(feature.capitalize())
    plt.legend(title='Audiogram Profile')
    plt.grid(True)
    
    # Save the graph
    plt.savefig(folder.joinpath(f'{feature}_graph.png'), bbox_inches='tight')
    plt.close()

def compare_instruments_for_feature(data: pd.DataFrame, feature: str, f0_mixture: str, folder: pathlib.Path) -> None:
    """
    Creates a bar graph comparing instruments for a given feature and f0 mixture ratio.

    Args:
        data: The DataFrame containing the data to plot
        feature: The feature to plot
        f0_mixture: The f0 mixture ratio to filter the data for

    Returns:
        None
    """
    if feature not in data.columns:
        raise ValueError(f"The feature '{feature}' does not exist in the DataFrame.")
    
    # Filter data for the specific f0 mixture ratio
    filtered_data = data[data['f0_ratio'] == f0_mixture]
    
    # Group by 'instrument' and 'audiogram_profile', then calculate mean feature values
    grouped_data = filtered_data.groupby(['instrument', 'audiogram_profile'])[feature].mean().reset_index()
    
    # Pivot the data to get 'instrument' as rows and 'audiogram_profile' as columns
    pivoted_data = grouped_data.pivot(index='instrument', columns='audiogram_profile', values=feature)
    
    # Reorder columns
    pivoted_data = pivoted_data.reindex(columns=audiogram_colors.keys())

    # Plot
    pivoted_data.plot(kind='bar', stacked=False, color=audiogram_colors.values())

    # Adding graph elements
    plt.title(f'Comparison of {feature} Across Instruments and Audiogram Profiles\nMixture Ratio: {f0_mixture}')
    plt.xlabel('Instrument')
    plt.xticks(rotation=45)  # Rotate instrument names for better readability
    plt.legend(title='Audiogram Profile')
    plt.grid(axis='y', linestyle='--')

    # Change "distorted electric guitar" to "distorted e.g." for better readability
    labels = [item.get_text() for item in plt.gca().get_xticklabels()]
    labels = [label.replace('distorted electric guitar', 'distorted e.g.') for label in labels]
    plt.gca().set_xticklabels(labels)

    # Save the graph
    plt.savefig(folder.joinpath(f'{feature}_instrument_comparison.png'), bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze results from Samplifi run.')
    parser.add_argument('--file', required=True, type=str, help='The analysis results file to parse')
    parser.add_argument('--omit-f0', action='store_true', help='Omit the f0 ratio data from the graphs')

    args = parser.parse_args()
    filepath = pathlib.Path(args.file)
    omit_f0 = args.omit_f0
    
    # Prepare output folder
    analysis_folder = pathlib.Path('./analysis')
    os.makedirs(analysis_folder, exist_ok=True)
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Create a dataframe for spectral features
    df_spectral = normalize_for_spectral(data)
    df_haaqi = normalize_for_haaqi(data)
    
    # Create tables for each spectral feature
    feature_tables = {}
    for feature in ['voiced_probabilities', 
                    'spectral_flatness', 
                    'harmonic_energy', 
                    'mfcc_similarity']:
        spectral_feature_table = create_table_for_feature(df_spectral, feature, analysis_folder)
        feature_tables[feature] = spectral_feature_table
        # print(f'Average {feature} value across dataset:')
        # print(spectral_feature_table)
    
    # Create tables for HAAQI scores
    haaqi_table = create_table_for_feature(df_haaqi, 'haaqi_scores', analysis_folder)
    # print('Average HAAQI scores across dataset:')
    # print(haaqi_table)

    # Create a new dataframe with scored data
    df_spectral_scored = score_data(df_spectral)

    if omit_f0:
        df_spectral_scored = df_spectral_scored[df_spectral_scored['f0_ratio'] != 'f0']
        df_haaqi = df_haaqi[df_haaqi['f0_ratio'] != 'f0']

    # Create graphs for each spectral feature
    feature_graphs = {}
    for feature in ['voiced_probabilities_score',
                    'spectral_flatness_score',
                    'harmonic_energy_score',
                    'mfcc_similarity']:
        create_graph_for_feature(df_spectral_scored, feature, analysis_folder)

    # Create graphs for HAAQI scores
    create_graph_for_feature(df_haaqi, 'haaqi_scores', analysis_folder)

    # Compare instruments for each spectral feature
    mixture = '0.5_mix'
    for feature in ['voiced_probabilities',
                    'spectral_flatness',
                    'harmonic_energy',
                    'mfcc_similarity']:
        compare_instruments_for_feature(df_spectral, feature, mixture, analysis_folder)
    
    # Compare instruments for HAAQI scores
    compare_instruments_for_feature(df_haaqi, 'haaqi_scores', mixture, analysis_folder)
    