### Config

template_file = 'Input template v4.xlsx'

chosen_llm_full = 'ft:gpt-4o-2024-08-06:dobolyilab:ht:A3rU60He'
chosen_llm_budget = 'ft:gpt-4o-mini-2024-07-18:dobolyilab:ht:A3rD3mRe'

chosen_classifier = 'microsoft/deberta-v3-large (fine-tuned)'
model_path = 'use_remote' # folder should be added to the app's main directory (e.g., 'microsoft_deberta-v3-large'; for remote, this should be 'use_remote')

###

import streamlit as st
import streamlit_ext as ste

from process import *

SMTP_HOST     = st.secrets['smtp']['host']
SMTP_PORT     = int(st.secrets['smtp'].get('port', 587))
SMTP_USER     = st.secrets['smtp']['user']
SMTP_PASS     = st.secrets['smtp']['password']

EMAIL_SUPPORT = st.secrets['email']['support']
EMAIL_FROM    = SMTP_USER
EMAIL_TO      = st.secrets['email']['to']
EMAIL_CC      = st.secrets['email']['cc']

st.set_page_config(
    page_title = 'Content Validity Tool',
    page_icon = '📋',
    initial_sidebar_state = 'collapsed') # layout = 'wide'

st.markdown('''<style>
            .steDownloadButton:hover {border-color: rgb(255,255,255) !important; background-image: linear-gradient(rgb(0 0 0/20%) 0 0) !important;} 
            #content-validity-tool {text-align: center; padding-top: 0px;} 
            .stProgress > div > div > div > div {background-color: #1f83e1;}
            a {text-decoration: none !important;}
            a:hover {text-decoration: underline !important;}
            </style>''', 
            unsafe_allow_html = True)

version_number = '1.0.0'

@st.cache_resource
def st_load_model (model_path):
    return load_model(model_path)

@st.cache_resource
def read_template (template_file):
    buffer = open('templates/' + template_file, 'rb')
    return buffer

with st.sidebar:

    with st.expander('**General Options**', expanded = False):
        readability_enabled = st.checkbox('Enable Readability Output',
                                          help = 'This feature enables Flesch Reading Ease scores as an output sheet.')
        
        dist_and_rep_enabled = st.checkbox('Enable Additional Distinctiveness / Representativeness Output (**Warning:** Experimental)',
                                           help = 'This feature is work-in-progress. Please only enable it if you know what you need it for. For more info on the distinctiveness calculations, see [Colquitt et al. (2019)](https://psycnet.apa.org/record/2019-17805-001); for representativeness, see [MacKenzie et al. (2011)](https://misq.umn.edu/construct-measurement-and-validation-procedures-in-mis-and-behavioral-research-integrating-new-and-existing-techniques.html).')
    
    with st.expander('**Model Selection**', expanded = False):
        selected_model = st.radio(
        'Selected Model',
        [
            '**RATERC:** Returns probability scores indicating how well items align semantically with construct definitions.', 
            '**RATERD:** Rates how well items measure construct definitions on a 1-to-7 ordinal scale (extremely poor to excellent) based on multiple synthetic raters.'
        ],
        index = 0,
        label_visibility = 'collapsed')
    
        if selected_model.startswith('**RATERC'):
            chosen_model = chosen_classifier

            alt_color_enabled = st.checkbox('Enable Alternate Colorization',
                                            help = 'This feature provides an alternate colorization scheme for highlighting items that match vs. mismatch intended definitions.')
            
            if alt_color_enabled:
                alt_color_upper = st.slider("Match Boundary Upper", 
                                    min_value = .500, 
                                    value = .620, 
                                    max_value = 1.000,
                                    format = '%.3f')
                alt_color_lower = st.slider("Match Boundary Lower", 
                                    min_value = 0.000, 
                                    value = .062, 
                                    max_value = .499,
                                    format = '%.3f')
            else:
                alt_color_upper, alt_color_lower = None, None

        elif selected_model.startswith('**RATERD'):
            st.write('**Model Options**')

            openai_api_key_demo = st.text_input('OpenAI API Key', 
                                                openai_api_key_demo, 
                                                help = 'For instructions on how to acquire a key, visit: https://help.openai.com/en/articles/4936850-where-do-i-find-my-openai-api-key', 
                                                disabled = False)
            
            if openai_api_key_demo != 'sk-proj-...':
                client = OpenAI(api_key = openai_api_key_demo, max_retries = 10)

            if 'disabled' not in st.session_state:
                st.session_state.disabled = False

            num_seeds = st.number_input("Number of Synthetic Raters", 
                                    min_value = 2, 
                                    value = 10, 
                                    max_value = 50, 
                                    disabled = st.session_state.disabled, 
                                    help = "The number of synthetic raters will determine the quality of the RATERD model's results, but a higher number of runs will incur higher costs (e.g., OpenAI API fees and/or time spent waiting for results). Setting this value to at least 10 is recommended.")

            llm_test_mode = st.checkbox('Use GPT-4o Model',
                                        help = 'Check this box to use the larger GPT-4o model rather than the smaller GPT-4o mini model.')
            
            if llm_test_mode:
                chosen_llm = chosen_llm_full
            else:
                chosen_llm = chosen_llm_budget

            chosen_model = chosen_llm

            alt_color_enabled, alt_color_upper, alt_color_lower = None, None, None

        st.write('**Model Details:** ' + chosen_model)

    with st.expander('**Version Details**', expanded = False):
        st.write('**{version_number} :** Subject to Change'.format(version_number = version_number))

with st.container(border = True):
    st.header(
        'Content Validity Tool', 
        divider = 'rainbow', 
        anchor = False)

    with st.expander('**General Introduction**', expanded = False):
        st.markdown(
'''This tool is intended to help researchers in the social sciences to build psychometric scales that adequately represent the entire content domain of the construct it aims to measure, ensuring that all relevant aspects are covered and irrelevant elements are excluded.
        
To use the tool, you'll need to provide your scale items (i.e., the text of each question) and a brief definition of the target construct. Including "orbiting scales" – conceptually related but distinct construct definitions/items – is recommended to establish the uniqueness of your scale.
        
The tool will return numerical scores reflecting how well each item aligns semantically with a construct definition, helping you evaluate each item's effectiveness in representing the target concept. The tool can also support item and definition refinement, enhance scale distinctiveness, or clarify the content domain of a construct.

The app allows you to choose among several models to generate these scores via the left sidebar. Each model is based on a different training dataset and architecture, though all have comparable accuracy levels. If you're new to content validation, we recommend starting with the RATER<sub>C</sub> (classifier) model, which aligns with the approach put forth by [Anderson and Gerbing (1991)](https://psycnet.apa.org/record/1992-03961-001). Alternatively, consider the RATER<sub>D</sub> (distribution) model, which simulates the [Hinkin and Tracey (1999)](https://journals.sagepub.com/doi/10.1177/109442819922004) approach.''',
unsafe_allow_html = True)
    
    with st.expander('**Prepare Your Instrument**', expanded = False):
        st.markdown('Instructions on how to format your instrument are provided in the template itself.')

        download2 = ste.download_button(
            label = "Download Sample Template",
            data = read_template(template_file),
            file_name = template_file,
            mime = 'application/vnd.ms-excel',
            custom_css = 'background-color: rgb(38,39,48) !important; color: rgb(255,255,255) !important;'
            )

    def flip_disabled_state ():
        if st.session_state['uploaded_file'] is None:
            st.session_state.disabled = False
        else:
            st.session_state.disabled = True

    with st.expander('**Analyze Your Instrument**', expanded = False):

        st.markdown('Upload your prepared instrument here. As soon as the instrument is uploaded, the analysis will start.')

        uploaded_file = st.file_uploader(label = 'Upload Instrument', 
                                        key = 'uploaded_file', 
                                        label_visibility = 'collapsed', 
                                        on_change = flip_disabled_state)
    if uploaded_file is not None:
        st.subheader('Results', 
                     divider = 'rainbow', 
                     anchor = False)
        
        if model_path != 'use_remote':
            model, pipe = st_load_model(model_path)
        else:
            pipe = None

        with st.spinner('Processing...'):
            classifier_out, classifier_raw, seed_dfs = None, None, None

            if selected_model.startswith('**RATERD') and openai_api_key_demo == 'sk-proj-...':
                st.warning('**:red[Error:] you must enter a valid (and funded) OpenAI API key in the sidebar to use a RATERD model. For instructions on acquiring a key, please see: https://help.openai.com/en/articles/4936850-where-do-i-find-my-openai-api-key**', icon = '⚠️')
            else:

                try:

                    template = organize_template(uploaded_file)

                    if selected_model.startswith('**RATERC'):
                        cls_seeds_bar = st.progress(0, text = None)
                        classifier_out, template = run_classifier(template, pipe, dist_and_rep_enabled)
                        cls_seeds_bar.progress(100, text = None)
                    elif selected_model.startswith('**RATERD'):
                        llm_seeds_bar = st.progress(0, text = None)

                        seed_nums = list(range(123, 123 + num_seeds))
                        seed_dfs = []
                        for i, each_seed in enumerate(seed_nums):
                            llm_seeds_bar.progress((i + 1) / len(seed_nums), text = None)
                            seed_dfs.append(run_llm(template, seed_num = each_seed, chosen_llm = chosen_llm, client_pass = client))

                    print([alt_color_enabled])
                    buffer = process_results(classifier_out, template, seed_dfs, chosen_model, readability_enabled, dist_and_rep_enabled, alt_color_enabled, alt_color_upper, alt_color_lower, 'buffer')

                    download3 = ste.download_button(
                        label = "Download Results",
                        data = buffer,
                        file_name = 'result.xlsx',
                        mime = 'application/vnd.ms-excel',
                        custom_css = 'background-color: rgb(34,139,34) !important; color: rgb(255,255,255) !important;'
                        )

                except Exception as e:
                    msg = str(e)
                    st.warning('**:red[Error:] Results could not be generated (see details below)! Please check your template for issues (or email ' + EMAIL_SUPPORT + ' for support).**', icon = '⚠️')
                    st.warning('**:red[Error Details:]** ' + msg, icon = '⚠️')

with st.container(border = True):
    with st.expander('**Citation**', expanded = True):
        st.markdown('Pillet, J. C., Larsen, K. R., Dobolyi, D., Queiroz, M., Handler, A., Arnulf, J. K., & Sharma, R. (2026). [AI-Augmented Content Validation in Behavioral Research: Development and Evaluation of the RATER System](https://doi.org/10.25300/MISQ/2025/18946). *MIS Quarterly*, *50*(1), 59-86. https://doi.org/10.25300/MISQ/2025/18946', unsafe_allow_html = False)

    import smtplib, ssl
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart
        
    with st.expander('**Feedback & Comments**', expanded = False):
        st.markdown('Please provide any feedback or comments regarding the tool below:',
unsafe_allow_html = False)

        user_email = st.text_input('Your Email (Optional)')
        feedback = st.text_area('Your Message')

        if st.button('Submit', type = 'secondary'):
            if feedback.strip():
                try:
                    msg = MIMEMultipart()
                    msg['From'] = EMAIL_FROM
                    msg['To'] = EMAIL_TO
                    msg['Cc'] = EMAIL_CC
                    msg['Subject'] = 'Feedback about contval.org'
                    
                    body = f"Feedback from: {user_email or 'Anonymous'}\n\n{feedback}"
                    msg.attach(MIMEText(body, 'plain'))

                    context = ssl.create_default_context()
                    with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout = 15) as server:
                        server.ehlo()
                        server.starttls(context = context)
                        server.ehlo()
                        server.login(SMTP_USER, SMTP_PASS)
                        server.sendmail(msg['From'], [msg['To']], msg.as_string())

                    st.success('✅ Thank you for your feedback! It has been sent.')

                except Exception as e:
                    st.error('❌ Could not send email: {e}')
            else:
                st.warning('Please enter some feedback before submitting.')