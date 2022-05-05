import os
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import f1_score, jaccard_score, precision_score, recall_score

def calculate_scores(y_true: np.array, y_pred: np.array, chip_id: str, m_options: dict) -> pd.DataFrame:
    '''Generates DataFrame with F1_score, Jaccard, Recall, Precision and additional information for current inference run'''

    score_df = pd.DataFrame(data=[[0,0,0,0,0,0,0,0,0,0,0,0]], columns=['Date', 'Chip_id', 'Model_name', 'TTA', 'Threshold', 'Post-Pro.', 'Iter.', 'Kernel_s.', 'Jaccard', 'F1_score', 'Precision', 'Recall'])

    # Add date & time
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S") # dd/mm/YY H:M:S
    score_df['Date'] = dt_string

    # Add chip_id
    score_df['Chip_id'] = chip_id

    # Add model_name
    if len(m_options['model_option']) > 1:
        # Shorten model names to fit them all into the column
        models_str = ', '.join(f'{model_n[:8]}...' for model_n in m_options['model_option'])
        score_df['Model_name'] = models_str
    else:
        score_df['Model_name'] = m_options['model_option'][0]

    # Add tta_option
    score_df['TTA'] = int(m_options['tta_option'])

    # Add tta_option
    score_df['Threshold'] = round(m_options['threshold_option'], 2)

    # Add post-processing
    score_df['Post-Pro.'] = m_options['pp_option'].split(' ')[1]
    score_df['Iter.'] = int(m_options['pp_iter_option'])
    score_df['Kernel_s.'] = int(m_options['pp_kernel_option'])

    # Calculate and add scores
    jaccard_sc = jaccard_score(y_true=y_true, y_pred=y_pred)
    score_df['Jaccard'] = jaccard_sc
    f1_sc = f1_score(y_true=y_true, y_pred=y_pred)
    score_df['F1_score'] = f1_sc
    recall_sc = recall_score(y_true=y_true, y_pred=y_pred)
    score_df['Recall'] = recall_sc
    precision_sc = precision_score(y_true=y_true, y_pred=y_pred)
    score_df['Precision'] = precision_sc

    score_df_path = 'temp/score_df.csv'

    if os.path.isdir('temp'):
        try:
            # Open score_df from temp folder and add new run on top
            temp_score_df = pd.read_csv(score_df_path)

            frames = [score_df, temp_score_df]
            score_df = pd.concat(frames)
            score_df.to_csv(score_df_path, index=False)
        except:
            score_df.to_csv(score_df_path, index=False)
    else:
        os.mkdir('temp')
        score_df.to_csv(score_df_path, index=False)
       
    score_df = score_df.reset_index(drop=True)

    return score_df