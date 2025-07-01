### Config

set_batch_size = 128

openai_api_key = '...' # add valid key
openai_api_key_demo = 'sk-proj-...'

set_max_workers = 100
set_temperature = 1

readability_threshold = 25

###

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from openai import OpenAI
client = OpenAI(api_key = openai_api_key, max_retries = 10)

from prompts import *

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
 
import pandas as pd
import numpy as np
from xlsxwriter.utility import xl_col_to_name

import textstat

import json
import io

import warnings
warnings.filterwarnings('ignore', message = 'The sentencepiece tokenizer that you are converting to a fast tokenizer')

def load_model (model_path = 'model'):
    import torch
    torch.classes.__path__ = []
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, TextClassificationPipeline

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = AutoModelForSequenceClassification.from_pretrained(
    model_path,
    num_labels = 2
    ).to(device)

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, clean_up_tokenization_spaces = True)
    except:
        config = json.load(open(model_path + '/config.json'))
        tokenizer = AutoTokenizer.from_pretrained(config['_name_or_path'], clean_up_tokenization_spaces = True)
        tokenizer.save_pretrained(model_path)

    pipe = TextClassificationPipeline(model = model, tokenizer = tokenizer, top_k = None, device = device)

    return(model, pipe)

def organize_template (file):
    defs = pd.read_excel(file, sheet_name = 'Definitions').fillna('')
    items = pd.read_excel(file, sheet_name = 'Items').fillna('')
    defs['Abbreviation'] = defs['Abbreviation'].str.strip()

    items['Expected target construct'] = items['Expected target construct'].replace('', '(blank)')
    #items = items.sort_values('Expected target construct', ascending = True)#.reset_index(drop = True)
    items = items.groupby('Expected target construct').apply(lambda x: x.sort_index()).droplevel(0)

    defs['ShortCode'] = defs['Abbreviation']
    items['ShortCode'] = items['Expected target construct']
    items['Code'] = range(1, len(items) + 1)

    data = items.merge(defs, how = 'cross', suffixes = ('', 'Definition'))
    data['ItemDefinitionCorrespondence'] = np.where(data['ShortCode'] == data['ShortCodeDefinition'], 'Match', 'Mismatch')
    data = data.sort_values(['Code', 'ItemDefinitionCorrespondence', 'ShortCode'], ascending = [True, True, True])
    data = data.drop(columns = ['ItemDefinitionCorrespondence'])
    data['Code'] = ['Item ' + str(x) for x in data['Code']]

    data['Readability'] = [textstat.flesch_reading_ease(x) for x in data['Definition'] + ' ' + data['Item statement']]
         
    return data

def process_llm_results (seed_dfs):
        all_data = pd.concat(seed_dfs)
        seed_nums = list(sorted(all_data['Seed'].unique()))

        structure = seed_dfs[0].drop(['LLM Scale Response', 'LLM Scale Response', 'Seed'], axis = 1)
        seed_data = all_data.pivot(columns = 'Seed',
                                values = 'LLM Scale Response')
        
        wide_seed_data = structure.join(seed_data)[['ShortCode', 'Code', 'Definition', 'ShortCodeDefinition', 'Reversed item', 'Item statement'] + list(seed_nums)]

        wide_seed_data = wide_seed_data.assign(Mean = wide_seed_data[seed_nums].mean(axis = 1))

        SCDs = sorted(wide_seed_data['ShortCodeDefinition'].unique())

        res = []
        for i, each in enumerate(wide_seed_data['Code'].unique()):
            res.append(wide_seed_data[wide_seed_data['Code'] == each].reset_index(drop = True))

        rows = []

        for each in res:
            item_statement = each['Item statement'].unique()[0]
            is_reversed = each['Reversed item'].unique()[0]

            work = each.drop(columns = ['Definition', 'Reversed item', 'Item statement']).sort_values('ShortCodeDefinition').reset_index(drop = True)

            predicted_target_construct = work.iloc[np.argmax(work['Mean'])]['ShortCodeDefinition']
            
            for i, each in enumerate(work.iterrows()):
                #short_code = work.loc[i,]['ShortCode']

                if work['ShortCode'].unique()[0] == '(blank)':
                    short_code = predicted_target_construct
                else:
                    short_code = work.loc[i,]['ShortCode']

                short_code_definition = work.loc[i,]['ShortCodeDefinition']
                work_no_codes = work.drop(columns = ['ShortCode', 'Code', 'ShortCodeDefinition'])

                if short_code == short_code_definition:
                    s_htc = np.round(work.loc[i,]['Mean'] / 7, 2)

                add_cols = work_no_codes.loc[i:i,].add_prefix(work.loc[i,]['ShortCodeDefinition'] + '_').reset_index(drop = True)
                
                if i == 0:
                    build_row = add_cols
                else:
                    build_row = build_row.join(add_cols)

            build_row['Predicted target construct'] = predicted_target_construct

            if work['ShortCode'].unique()[0] == '(blank)':
                matched_mean = work.query('ShortCodeDefinition == "' + build_row['Predicted target construct'][0] + '"')['Mean']
                alternate_means = work.query('ShortCodeDefinition != "' + build_row['Predicted target construct'][0] + '"')['Mean']            
            else:
                matched_mean = work.query('ShortCode == ShortCodeDefinition')['Mean']
                alternate_means = work.query('ShortCode != ShortCodeDefinition')['Mean']

            s_htd = np.round(np.mean(matched_mean.to_numpy() - alternate_means.to_numpy()) / (7 - 1), 2)

            build_row['s_htc'] = s_htc
            build_row['s_htd'] = s_htd

            build_row['Expected target construct'] = work['ShortCode'].unique()
            build_row['Scoring target construct'] = short_code
            build_row['Reversed item'] = is_reversed
            build_row['Item statement'] = item_statement
            
            rows.append(build_row)

        out = pd.concat(rows)

        out.insert(0, 'Item statement', out.pop('Item statement'))
        out.insert(1, 'Expected target construct', out.pop('Expected target construct'))
        out.insert(2, 'Predicted target construct', out.pop('Predicted target construct'))
        out.insert(3, 'Scoring target construct', out.pop('Scoring target construct'))
        out.insert(4, 'Reversed item', out.pop('Reversed item'))
        out.insert(5, 's_htc', out.pop('s_htc'))
        out.insert(6, 's_htd', out.pop('s_htd'))

        out = out.sort_values('Scoring target construct', ascending = True).reset_index(drop = True)

        #print(out)

        return out, SCDs, seed_nums

def process_results (classifier_out, classifier_raw, seed_dfs, chosen_model, readability_enabled, dist_and_rep_enabled, alt_color_enabled, alt_color_upper, alt_color_lower, output = 'buffer'):

    if seed_dfs is not None:
        out, SCDs, seed_nums = process_llm_results(seed_dfs)

    definitions = classifier_raw[['Construct name', 'Abbreviation', 'Definition']].drop_duplicates().sort_values('Abbreviation')

    buffer = io.BytesIO()

    if output == 'buffer':
        excel_path = buffer
    else:
        excel_path = 'out.xlsx'

    with pd.ExcelWriter(excel_path, engine = 'xlsxwriter') as writer:

        def mismatch_colorizer (df, worksheet, formulas = None, values = None):
            mismatches = np.where((df['Expected target construct'] != '(blank)') & (df['Expected target construct'] != df['Predicted target construct']))[0]

            for each_mismatch in mismatches:

                if formulas is not None:
                    worksheet.write_formula(xl_col_to_name(df.columns.get_loc('Predicted target construct')) + str(each_mismatch + 2), formulas[each_mismatch], value = values[each_mismatch], cell_format = highlight_mismatch)
                else:
                    worksheet.write(xl_col_to_name(df.columns.get_loc('Predicted target construct')) + str(each_mismatch + 2), df['Predicted target construct'][each_mismatch], highlight_mismatch)

        def calc_predicted (df, start_col, end_col, PTC_idx):
            formulas = []
            values = []
            for index, row in df.iterrows():
                row_num = str(index + 2)

                formula = '=INDEX($' + start_col + '$1:$' + end_col + '$1,MATCH(MAX($' + start_col + '$' + row_num + ':$' + end_col + '$' + row_num + '),$' + start_col + '$' + row_num + ':$' + end_col + '$' + row_num + ',0))'
                value = row.iloc[PTC_idx]

                worksheet.write_formula(xl_col_to_name(PTC_idx) + row_num, formula, value = value)

                formulas.append(formula)
                values.append(value)

            return formulas, values

        def calc_r (df, calc_cols):

            options_representativeness = {
                    'width': 445,
                    'height': 350,
                    'font': {'name': 'Lucida Sans Typewriter',
                             'color': 'white',
                             'size': 10},
                    'align': {'vertical': 'top',
                              'horizontal': 'left'},
                    'fill': {'color': '#262626'},
                }

            total_dims = len(calc_cols)
            options_representativeness['height'] = 200 + 35 * total_dims

            #print(options_representativeness['height'])
            
            calc_representativeness_text = '''******************\nRepresentativeness\n******************\n\n'''

            all_exp_vals = []
            tc_names = [cols[x] for x in calc_cols]
            all_cols = [xl_col_to_name(x) for x in calc_cols]

            for index, each_col in enumerate(tc_names):
                exp_sum = np.exp(np.sum(df[each_col]))
                all_exp_vals.append(exp_sum)
                calc_representativeness_text += 'Exp Sum of ' + each_col + ': EXP(SUM(' + all_cols[index] + '2:' + all_cols[index] + str(len(df) + 1) + ')) = ' + str(np.round(exp_sum, 3)).lstrip('0') + '\n'

            calc_representativeness_text += '\nSum of exps: ' + str(np.round(np.sum(all_exp_vals), 3)) + '\n\n-----------------\nFinal Calculation\n-----------------\n\n'

            for index, each_col in enumerate(tc_names):
                calc_representativeness_text += 'Softmax of ' + each_col + ': ' + str(np.round(all_exp_vals[index], 3)) + '/' + str(np.round(np.sum(all_exp_vals), 3)) + ' = ' + str(np.round(all_exp_vals[index] / np.sum(all_exp_vals), 3)).lstrip('0') + '\n'

            worksheet.insert_textbox(len(df) + 2, cols.get_loc('Reversed item'), calc_representativeness_text, options_representativeness)

            return
        
        def calc_d_classifier (df, calc_cols, ETC_idx):

            for index, row in df.iterrows():

                if row.iloc[ETC_idx] != '(blank)':
                    etc_col = xl_col_to_name(cols.get_loc(row.iloc[ETC_idx]))
                    all_cols = [xl_col_to_name(x) for x in calc_cols]
                    non_etc_cols = all_cols.copy()
                    non_etc_cols.remove(etc_col)

                    row_num = str(index + 2)

                    formula = '=('

                    for each_col in non_etc_cols:
                        formula += '(' + etc_col + row_num + '-' + each_col + row_num + ') + '
                    formula = formula[:-3]
                    formula += ')/' + str(len(non_etc_cols))

                    worksheet.write_formula(xl_col_to_name(cols.get_loc('Distinctiveness')) + row_num, formula, value = row['Distinctiveness'])

            return

        workbook = writer.book

        # establish formatting
        highlight_mismatch = workbook.add_format({#'bg_color': 'red', 
                                #'font_color': 'white',
                                'bold': True})
        highlight_best_pred = workbook.add_format({'bg_color': 'black', 
                                'font_color': 'white',
                                'bold': True})
        highlight_warning = workbook.add_format({'bg_color': 'red', 
                                'font_color': 'white',
                                'bold': True})

        center_col = workbook.add_format({'align': 'center'})
        make_bold = workbook.add_format({'bold': True})

        round_2_digits_with_leading_0 = workbook.add_format({'num_format': '#,##0.00',
                                                             'align': 'center'})
        round_3_digits_no_leading_0 = workbook.add_format({'num_format': '###.000', 
                                                           'align': 'center'})
        round_integer = workbook.add_format({'num_format': '#,##0',
                                             'align': 'center'})

        # write instructions
        worksheet = workbook.add_worksheet('Instructions')

        if classifier_out is not None:
            text = open('templates/Output template instructions - Classifier.txt', 'r').read()
            text_height = 600
        elif seed_dfs is not None:
            text = open('templates/Output template instructions - Distribution.txt', 'r').read()
            text_height = 800

        options = {
            'width': 800,
            'height': text_height,
            'x_offset': 10,
            'y_offset': 10,
            'font': {'color': 'white',
                     'size': 12},
            'align': {'vertical': 'top',
                      'horizontal': 'left'},
            'fill': {'color': '#262626'},
        }

        worksheet.insert_textbox('A1', text, options)

        # if using classifier
        if classifier_out is not None:
            # write classifier calculations
            classifier_out = classifier_out.drop(columns = 'Code')
            classifier_out.to_excel(writer, sheet_name = 'Classifier Calculations', index = False)
            worksheet = writer.sheets['Classifier Calculations']

            (max_row, max_col) = classifier_out.shape
            cols = classifier_out.columns
            ETC_idx = cols.get_loc('Expected target construct')
            PTC_idx = cols.get_loc('Predicted target construct')
            RI_idx = cols.get_loc('Reversed item')
            start_col = xl_col_to_name(RI_idx + 1)
            end_col = xl_col_to_name(RI_idx + len(definitions['Abbreviation']))

            #worksheet.conditional_format(1, RI_idx + 1, max_row, max_col, {'type': '3_color_scale', 'min_value': 0, 'max_value': 1})

            worksheet.set_column(RI_idx, RI_idx, cell_format = center_col)
            worksheet.set_column(RI_idx + 1, max_col, cell_format = round_3_digits_no_leading_0)

            border_format_top = workbook.add_format({'top': 5, 'left': 5, 'right': 5})
            border_format_sides = workbook.add_format({'left': 5, 'right': 5})
            border_format_bottom = workbook.add_format({'left': 5, 'right': 5, 'bottom': 5})
            border_format_all = workbook.add_format({'border': 5})

            above_lower_threshold_any = workbook.add_format({'bg_color': 'yellow'})
            above_upper_threshold_match = workbook.add_format({'bg_color': 'black', 'font_color': 'white'})
            below_upper_threshold_match = workbook.add_format({'bg_color': 'red'})

            if alt_color_enabled:

                for j in range(RI_idx + 1, RI_idx + len(definitions['Abbreviation']) + 1):

                    print(j)

                    for i in range(0, max_row):
                        if classifier_out.iloc[i, j] >= alt_color_upper and classifier_out['Expected target construct'][i] == classifier_out.columns[j]:
                            apply_format = above_upper_threshold_match
                            apply = True
                        elif classifier_out.iloc[i, j] < alt_color_upper and classifier_out['Expected target construct'][i] == classifier_out.columns[j]:
                            apply_format = below_upper_threshold_match
                            apply = True
                        elif classifier_out.iloc[i, j] > alt_color_lower:
                            apply_format = above_lower_threshold_any
                            apply = True
                        else:
                            apply = False

                        if apply:
                            worksheet.conditional_format(i + 1, j, i + 1, j, {'type': 'no_blanks', 'format': apply_format})

                        print(i)

                    matched_cells = np.where((classifier_out['Expected target construct'] != '(blank)') & (classifier_out['Expected target construct'] == classifier_out.columns[j]))
                    if len(matched_cells[0]) >= 2:
                        top = matched_cells[0][0] + 1
                        bottom = matched_cells[0][-1] + 1
                        worksheet.conditional_format(top, j, top, j, {'type': 'no_blanks' , 'format': border_format_top})
                        worksheet.conditional_format(top, j, bottom - 1, j, {'type': 'no_blanks' , 'format': border_format_sides})
                        worksheet.conditional_format(bottom, j, bottom, j, {'type': 'no_blanks' , 'format': border_format_bottom})
                    elif len(matched_cells[0]) == 1:
                        top = matched_cells[0][0] + 1
                        worksheet.conditional_format(top, j, top, j, {'type': 'no_blanks' , 'format': border_format_all})
            else:
                for index, row in classifier_out.iterrows():
                    worksheet.conditional_format(index + 1, RI_idx + 1, index + 1, RI_idx + len(definitions['Abbreviation']), {'type': 'top', 'value': 1, 'format': highlight_best_pred})

            formulas, values = calc_predicted(classifier_out, start_col, end_col, PTC_idx)
            mismatch_colorizer(classifier_out, worksheet, formulas, values)

            # distinctiveness and representativeness
            if dist_and_rep_enabled:
                calc_d_classifier(classifier_out, range(RI_idx + 1, RI_idx + len(definitions['Abbreviation']) + 1), ETC_idx)
                calc_r(classifier_out, range(RI_idx + 1, RI_idx + len(definitions['Abbreviation']) + 1))

            worksheet.autofit()

            # write classifier predictions
            current_sheet = 'Classifier Predictions'
            classifier_raw.drop(columns = ['ShortCode', 'Code', 'ShortCodeDefinition']).to_excel(writer, sheet_name = current_sheet, index = False)
            writer.sheets[current_sheet].autofit()
            writer.sheets[current_sheet].hide()

        # if using LLM
        if seed_dfs is not None:
            # write LLM calculations
            out.to_excel(writer, sheet_name = 'Distribution Calculations', index = False)
            worksheet = writer.sheets['Distribution Calculations']

            (max_row, max_col) = out.shape
            worksheet.conditional_format(1, out.columns.get_loc('s_htc'), max_row, out.columns.get_loc('s_htc'), {'type': '3_color_scale', 'min_value': 1/7, 'max_value': 1, 'min_type': 'num', 'max_type': 'num'})
            worksheet.conditional_format(1, out.columns.get_loc('s_htd'), max_row, out.columns.get_loc('s_htd'), {'type': '3_color_scale', 'min_value': -1, 'max_value': 1, 'min_type': 'num', 'max_type': 'num'})

            cols = out.columns
            cols_idx = np.array(range(len(cols)))
            ETC_idx = cols.get_loc('Expected target construct')

            SCD_idx = cols_idx[np.array([x.endswith(str(seed_nums[0])) for x in cols])]
            Mean_idx = cols_idx[np.array([x.endswith('_Mean') for x in cols])]
            s_htc_idx = xl_col_to_name(cols.get_loc('s_htc'))
            s_htd_idx = xl_col_to_name(cols.get_loc('s_htd'))

            RI_idx = cols.get_loc('Reversed item')

            worksheet.set_column(RI_idx, RI_idx, cell_format = center_col)

            mismatch_colorizer(out, worksheet)

            for i, each in enumerate(SCD_idx):

                current_SCD = SCDs[i]
                start_col = xl_col_to_name(SCD_idx[i])
                end_col = xl_col_to_name(Mean_idx[i] - 1)
                end_col_grouping = xl_col_to_name(Mean_idx[i])
                mean_col = xl_col_to_name(Mean_idx[i])
                other_mean_cols = [xl_col_to_name(x) for x in np.delete(np.array(Mean_idx), i)]

                for index, row in out.iterrows():
                    row_num = str(index + 2)
                    worksheet.write_formula(mean_col + row_num, '=AVERAGE(' + start_col + row_num + ':' + end_col + row_num + ')', round_2_digits_with_leading_0)

                    if row['Scoring target construct'] == current_SCD:
                        # s_htd
                        formula = ''
                        for each_other_mean_col in other_mean_cols:
                            formula += mean_col + row_num + '-' + each_other_mean_col + row_num + ', '

                        worksheet.write_formula(s_htd_idx + row_num, '=AVERAGE(' + formula[:-2] + ')/(7-1)', round_2_digits_with_leading_0)

                worksheet.set_column(start_col + ':' + end_col_grouping, None, None, {'level': 1, 'hidden': True})
                worksheet.set_column(end_col_grouping + ':' + end_col_grouping, None, None, {'collapsed': True})

            def write_s_htc_formulae (input, mean_col):
                idxs = out.index[input] + 2
                idxs_fl = [idxs[0], idxs[-1]]

                formula = ':'.join([mean_col + str(x) for x in idxs_fl]) + '/7'
                worksheet.write_dynamic_array_formula(':'.join([s_htc_idx + str(x) for x in idxs_fl]), formula, round_2_digits_with_leading_0)

            # s_htc
            for i, each_SCD in enumerate(SCDs):
                if each_SCD in out['Scoring target construct'].unique():
                    write_s_htc_formulae(out['Scoring target construct'] == each_SCD, xl_col_to_name(Mean_idx[i]))

            # distinctiveness and representativeness
            if dist_and_rep_enabled:
                calc_r(out, Mean_idx)

            # write seeds
            for each_seed in seed_dfs:
                current_sheet = 'LLM Seed ' + str(each_seed['Seed'].unique()[0])
                each_seed.insert(0, 'Seed', each_seed.pop('Seed'))

                each_seed.drop(columns = ['ShortCode', 'Code', 'ShortCodeDefinition']).to_excel(writer, sheet_name = current_sheet, index = False)

                writer.sheets[current_sheet].autofit()
                writer.sheets[current_sheet].hide()

            worksheet.autofit()

        # readability
        if readability_enabled:
            current_sheet = 'Readability'
            readability = classifier_raw[['Abbreviation', 'Definition', 'Item statement', 'Readability']]
            readability.to_excel(writer, sheet_name = current_sheet, index = False)

            worksheet = writer.sheets[current_sheet]
            (max_row, max_col) = readability.shape
            readability_loc = readability.columns.get_loc('Readability')

            worksheet.set_column(readability_loc, readability_loc, cell_format = round_integer)
            worksheet.conditional_format(1, readability_loc, max_row, readability_loc, {'type': 'cell', 'criteria': 'less than', 'value': readability_threshold, 'format': highlight_warning})

            worksheet.write_comment(xl_col_to_name(readability_loc) + '1', 'Scores are flagged if Flesch Reading Ease < ' + str(readability_threshold) + '.')

            writer.sheets[current_sheet].autofit()

        # write definitions
        current_sheet = 'Definitions'
        definitions.to_excel(writer, sheet_name = current_sheet, index = False)
        writer.sheets[current_sheet].autofit()

        # write model details
        current_sheet = 'Model Details'
        model_details = pd.DataFrame({'Model': chosen_model,
                                      'Run Date': pd.Timestamp.today()},
                                      index = [0])
        print(model_details)
        model_details.to_excel(writer, sheet_name = current_sheet, index = False)
        writer.sheets[current_sheet].autofit()
        writer.sheets[current_sheet].hide()

        #writer.close()

        print('Run finished!')

        return buffer
    
def answer_prompt (person, prompt, temperature, seed, chosen_llm):
    chat_response = client.chat.completions.create(
        model = chosen_llm, 
        messages=[
            {"role": "system", "content": person},
            {"role": "user", "content": prompt}
        ],
        temperature = temperature,
        max_tokens = 128,
        seed = seed,
        extra_body = {},
        timeout = 5,
    )

    return chat_response.choices[0].message.content

def run_inference (each_prompt, each_seed, chosen_llm):
    try:
        res = answer_prompt(person = thePerson, 
                            prompt = each_prompt,
                            temperature = set_temperature,
                            seed = each_seed,
                            chosen_llm = chosen_llm)
    except:
        print('Prompt processing error...')
    return(res)

def run_llm (file, seed_num, chosen_llm):
    prompts = []

    for index, row in file.iterrows():
        prompt = thePrompt.format(statement = row['Definition'], item = row['Item statement'])
        prompts += [prompt]

    print('Evaluating Seed:', seed_num)

    with ProcessPoolExecutor(max_workers = np.min([len(prompts), set_max_workers])) as executor:
        res = list(executor.map(run_inference, prompts, [seed_num] * len(prompts), [chosen_llm] * len(prompts)))

    try:
        pred = [int(res[0]) for res in res]
    except:
        print('### NOTE: Issue with integer coding -- manual recode required! ###')
        pred = recode_preds(res)

    seed_df = file.copy()
    seed_df['LLM Response'] = res
    seed_df['LLM Scale Response'] = pred
    seed_df['Seed'] = seed_num

    return seed_df

def recode_preds (preds):
    categories = ['EXTREMELY BAD', 'VERY BAD', 'SOMEWHAT BAD', 'ADEQUATE', 'SOMEWHAT GOOD', 'VERY GOOD', 'EXTREMELY GOOD']

    recodes = []

    for pred in preds:
        try:
            recodes.append(int(pred[0]))
        except:
            print('---')
            print('Problematic Response:', pred)
            replacement = ''
            for i, each_category in enumerate(categories):
                if pred.find(each_category) != -1:
                    replacement = i + 1
            if replacement == '':
                replacement = 1
            
            recodes.append(replacement)
            print('Recode:', replacement)
    
    print('---')

    return recodes

def run_classifier (template, pipe = None, dist_and_rep_enabled = False):
    print('Running classifier...')

    template_dict = []

    for i, _ in template.iterrows():
        template_dict.append({'text': template['Definition'][i], 
                              'text_pair': template['Item statement'][i]})

    if pipe is None:
        model, pipe = load_model(model_path)

    raw_probs = pipe(template_dict, batch_size = 16)
    
    probs = np.array([item[['LABEL_1' == i['label'] for i in item].index(True)]['score'] for item in raw_probs])

    template['Classifier Probabilities'] = probs

    classifier_out_wide = template.pivot(index = 'Code', 
                               columns = 'ShortCodeDefinition', 
                               values = 'Classifier Probabilities').reset_index().rename_axis(None, axis = 1).assign()

    best_match = []

    for index, row in classifier_out_wide.iterrows():
        best_match.append(classifier_out_wide.columns[1:][np.argmax(row[1:])])

    classifier_out_wide['Predicted target construct'] = best_match
    classifier_out_wide.insert(1, 'Predicted target construct', classifier_out_wide.pop('Predicted target construct'))

    classifier_out = template[['Code', 'Expected target construct', 'Reversed item']].drop_duplicates().reset_index(drop = True).merge(classifier_out_wide)

    classifier_out.insert(3, 'Reversed item', classifier_out.pop('Reversed item'))

    def ETCProb (x):
        if x['Expected target construct'] != '(blank)':
            return x[x['Expected target construct']]
        else:
            return np.nan

    classifier_out['Probability for expected target construct'] = classifier_out.apply(lambda x: ETCProb(x), axis = 1)

    if dist_and_rep_enabled:
        cols = classifier_out.columns
        distinctiveness = []
        for index, row in classifier_out.iterrows():
            all_tc_cols = list(range(cols.get_loc('Reversed item') + 1, cols.get_loc('Probability for expected target construct')))
            if row['Expected target construct'] != '(blank)':
                etc_idx = cols.get_loc(row['Expected target construct'])
                all_tc_cols.remove(etc_idx)
                distinctiveness.append(np.sum(row.iloc[etc_idx] - row.iloc[all_tc_cols]) / len(all_tc_cols))
            else:
                distinctiveness.append(np.nan)
        classifier_out['Distinctiveness'] = distinctiveness

    classifier_out = classifier_out.merge(template[['Code', 'Item statement']].drop_duplicates().reset_index(drop = True), how = 'left', on = 'Code')
    classifier_out.insert(0, 'Item statement', classifier_out.pop('Item statement'))

    return classifier_out, template

if __name__ == "__main__":
    template = organize_template('templates/Input template v2.xlsx')
    print('### Running Classifier ###')
    classifier_out, classifier_raw = run_classifier(template)
    print('### Running LLM ###')
    seed_dfs = run_llm(template, num_seeds = 2)
    buffer = process_results(classifier_out, classifier_raw, seed_dfs, True, True, True, .062, .620, 'local')