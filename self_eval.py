# Required imports for application functionality
import os
import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import faiss
from openai import OpenAI
import tiktoken
from langchain_community.llms import OpenAI as LangChainOpenAI
from openpyxl import load_workbook

# Configure Streamlit page settings - MUST BE FIRST!
st.set_page_config(page_title="Self-Eval Assistant", page_icon="", layout="wide")

# Initialize session state variables
if 'accepted_terms' not in st.session_state:
    st.session_state.accepted_terms = False
if 'data_processed' not in st.session_state:
    st.session_state.data_processed = False
if 'embeddings_created' not in st.session_state:
    st.session_state.embeddings_created = False
if 'index_ready' not in st.session_state:
    st.session_state.index_ready = False
if 'api_key_valid' not in st.session_state:
    st.session_state.api_key_valid = False
if 'documents' not in st.session_state:
    st.session_state.documents = []
if 'embeddings' not in st.session_state:
    st.session_state.embeddings = None
if 'faiss_index' not in st.session_state:
    st.session_state.faiss_index = None
if 'nlg_template' not in st.session_state:
    st.session_state.nlg_template = None

# Display warning page for first-time users
if not st.session_state.accepted_terms:
    st.markdown("""
        <style>
        .warning-header {
            color: white;
            text-align: center;
            padding: 20px;
            margin-bottom: 20px;
        }
        .warning-section {
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 15px;
            border-left: 4px solid #ff4b4b;
            color: black;
        }
        </style>
        """, unsafe_allow_html=True)
    
    st.markdown("<h1 class='warning-header'>Welcome to Self-Eval Assistant</h1>", unsafe_allow_html=True)
    
    st.markdown("<div class='warning-section'>", unsafe_allow_html=True)
    st.markdown("### 1. Data Security")
    st.markdown("""
    - Your performance data remains confidential
    - Secure API key handling
    - Local data processing when possible
    - Regular security updates
    """)
    st.markdown("</div>", unsafe_allow_html=True)
    
    agree = st.checkbox("I understand and agree to the data handling policies")
    if st.button("Continue to Self-Eval Assistant", disabled=not agree):
        st.session_state.accepted_terms = True
        st.rerun()
    st.stop()

# Function to process evaluation data and create embeddings
def process_evaluation_data(df):
    try:
        client = OpenAI(api_key=st.session_state.api_key)
        text_data = df.to_string()
        response = client.embeddings.create(
            input=text_data,
            model="text-embedding-ada-002"
        )
        return np.array([response.data[0].embedding])
    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        return None

# Function to generate evaluation analysis
def generate_evaluation_analysis(df, part_number):
    try:
        achievements_text = df.to_string()
        
        prompts = {
            1: f"""
            Based on the following Strategic Objectives (SOs) and accomplishments:
            {achievements_text}

            Provide a detailed evaluation for each SO category with specific justification points and individual scores.

            For each SO, provide:
            1. Bullet points of specific achievements
            2. Concrete examples and metrics where available
            3. Individual score based on achievement level:
            - [5] Exceeded all objectives with demonstrable above-target results
            - [4] Fully achieved all objectives
            - [3] Achieved most objectives
            - [2] Achieved some objectives
            - [1] Achieved minimal objectives
            - [0] No substantial achievement

            Format your response exactly as follows:

            SO#1 - UPSKILLING & TEAM CONTRIBUTION
            JUSTIFICATION:
            • [Achievement point with specific example]
            • [Achievement point with specific example]
            • [Achievement point with specific example]
            SCORE: [X/5]
            REASONING: [Brief explanation of score]

            SO#2 - QUALITY OUTPUT
            JUSTIFICATION:
            • [Achievement point with specific example]
            • [Achievement point with specific example]
            • [Achievement point with specific example]
            SCORE: [X/5]
            REASONING: [Brief explanation of score]

            SO#3 - JOB KNOWLEDGE
            JUSTIFICATION:
            • [Achievement point with specific example]
            • [Achievement point with specific example]
            • [Achievement point with specific example]
            SCORE: [X/5]
            REASONING: [Brief explanation of score]

            SO#4 - SPEED & ACCURACY
            JUSTIFICATION:
            • [Achievement point with specific example]
            • [Achievement point with specific example]
            • [Achievement point with specific example]
            SCORE: [X/5]
            REASONING: [Brief explanation of score]

            SO#5 - COST EFFICIENCY
            JUSTIFICATION:
            • [Achievement point with specific example]
            • [Achievement point with specific example]
            • [Achievement point with specific example]
            SCORE: [X/5]
            REASONING: [Brief explanation of score]

            OVERALL RATING: [Average of all SO scores]/5
            FINAL ASSESSMENT: [Brief overall performance summary]
            """,
            
            2: f"""
            Based on the following accomplishments:
            {achievements_text}

            Evaluate the behavioral competencies with specific examples and scoring:

            PROVIDES SUPPORT AND HELP TO OTHERS
            • Help others accomplish tasks/goals
            • Show proactiveness in task management
            • Volunteer for ad-hoc tasks
            • Peer feedback on teamwork
            Score: [5] External impact, [4] Department-wide, [3] Project-wide, [2] Team-wide, [1] Individual, [0] No care

            RESPECT
            • Control of emotions in high-pressure situations
            • Ethical behavior
            • Professional conduct
            • Respecting others' time
            Score: [5] Consistently Exceeds, [4] Often Exceeds, [3] Meets, [2] Needs Improvement, [1] Not Meeting, [0] Cultural unfit

            TRUST
            • Accepting/learning from mistakes
            • Acting in team's best interest
            • Professional integrity
            • Work transparency
            Score: [5] Consistently Exceeds, [4] Often Exceeds, [3] Meets, [2] Needs Improvement, [1] Not Meeting, [0] Trust issue

            EXCEED CUSTOMER EXPECTATIONS
            • Customer-centric problem solving
            • Thorough problem analysis
            • Win-win conflict management
            Score: [5] Consistently Exceeds, [4] Often Exceeds, [3] Meets, [2] Needs Improvement, [1] Not Meeting, [0] Don't care

            INITIATIVE
            • Fresh ideas and alignment with company mission
            • Productive use of idle time
            • Taking on challenging tasks
            • Proactive problem prevention
            • Self-development
            Score: [5] Consistently Exceeds, [4] Often Exceeds, [3] Meets, [2] Needs Improvement, [1] Not Meeting, [0] No initiative

            CORPORATE RESPONSIBILITY
            • Volunteer efforts
            • Promoting environmental policies
            Score: [5] Active participation, [4] Sometimes participates, [3] Willing to participate

            Format response with JUSTIFICATION, SCORE, and REASONING for each competency.
            """,
            
            3: f"""
            Based on the following accomplishments:
            {achievements_text}

            Evaluate innovative contributions in these areas:

            1. PROCESS IMPROVEMENTS
            Evaluation Criteria:
            • Efforts to seek "incremental" or "breakthrough" improvements
            • Evaluation and improvement of delivery processes for efficiency
            • Effectiveness and flexibility improvements
            Score:
            [5] - Proposed improvement has been implemented company wide and greatly optimized business operation
            [4] - Proposed improvement has been implemented department wide and optimized business operation
            [3] - Constantly share process improvement ideas within department at conceptual level
            [2] - Share process improvement ideas within team level from time to time
            [1] - No known instance of sharing improvement ideas

            2. NEW INNOVATIONS
            Evaluation Criteria:
            • Business innovation that improves existing products/services/processes
            • Solutions that create new customer value
            • Innovation that drives revenue through existing segments
            • Improvements to productivity or performance
            Score:
            [5] - Full implementation
            [4] - Created prototype and testable product
            [3] - Identify valuable and viable ideas
            [2] - Ideation
            [1] - No known instance of sharing innovation proposal

            Format your response exactly as follows:

            PROCESS IMPROVEMENTS
            JUSTIFICATION:
            • [Specific example of improvement implementation]
            • [Evidence of process evaluation]
            • [Impact on efficiency/effectiveness]
            SCORE: [X/5]
            REASONING: [Brief explanation of score based on implementation level]

            NEW INNOVATIONS
            JUSTIFICATION:
            • [Specific example of business innovation]
            • [Evidence of value creation]
            • [Impact on revenue/efficiency]
            SCORE: [X/5]
            REASONING: [Brief explanation of score based on implementation stage]

            OVERALL INNOVATION RATING: [Average of both scores]/5
            FINAL ASSESSMENT: [Brief summary of innovation contributions and potential]
            """
        }

        client = OpenAI(api_key=st.session_state.api_key)
        chat = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": """You are an expert performance evaluator.
                For each Strategic Objective:
                - Provide specific, measurable achievements
                - Include concrete examples and metrics
                - Score strictly based on evidence
                - Justify each score with clear reasoning
                Be objective and thorough in your evaluation."""},
                {"role": "user", "content": prompts.get(part_number, "Invalid part number")}
            ],
            temperature=0.7,
            max_tokens=2000
        )

        # Update the UI to display each SO separately
        response = chat.choices[0].message.content
        sections = response.split("\n\n")
        
        for section in sections:
            # For Part 1: Strategic Objectives
            if section.startswith("SO#"):
                st.markdown(f"### {section.split('\n')[0]}")
                
                justification_start = section.find("JUSTIFICATION:") + 14
                justification_end = section.find("SCORE:")
                justification = section[justification_start:justification_end].strip()
                st.markdown(justification)
                
                # Updated score styling with black text
                score_line = section[section.find("SCORE:"):].split("\n")[0]
                st.markdown(f"""
                <div style='background-color: #f0f2f6; padding: 10px; border-radius: 5px; width: fit-content;'>
                <span style='color: black;'><strong>{score_line}</strong></span>
                </div>
                """, unsafe_allow_html=True)
                
                reasoning_start = section.find("REASONING:") + 10
                reasoning = section[reasoning_start:].strip()
                st.markdown(f"**Reasoning:**\n{reasoning}")
                
                st.markdown("---")
            
            # For Part 2: Behavioral Competencies
            elif any(section.startswith(comp) for comp in ["PROVIDES SUPPORT", "RESPECT", "TRUST", "EXCEED CUSTOMER", "INITIATIVE", "CORPORATE"]):
                st.markdown(f"### {section.split('\n')[0]}")
                
                justification_start = section.find("JUSTIFICATION:") + 14
                justification_end = section.find("SCORE:")
                justification = section[justification_start:justification_end].strip()
                st.markdown(justification)
                
                score_line = section[section.find("SCORE:"):].split("\n")[0]
                st.markdown(f"""
                <div style='background-color: #f0f2f6; padding: 10px; border-radius: 5px; width: fit-content;'>
                <span style='color: black;'><strong>{score_line}</strong></span>
                </div>
                """, unsafe_allow_html=True)
                
                reasoning_start = section.find("REASONING:") + 10
                reasoning = section[reasoning_start:].strip()
                st.markdown(f"**Reasoning:**\n{reasoning}")
                
                st.markdown("---")
            
            # For Part 3: Innovation
            elif section.startswith(("PROCESS IMPROVEMENTS", "NEW INNOVATIONS")):
                st.markdown(f"### {section.split('\n')[0]}")
                
                justification_start = section.find("JUSTIFICATION:") + 14
                justification_end = section.find("SCORE:")
                justification = section[justification_start:justification_end].strip()
                st.markdown(justification)
                
                score_line = section[section.find("SCORE:"):].split("\n")[0]
                st.markdown(f"""
                <div style='background-color: #f0f2f6; padding: 10px; border-radius: 5px; width: fit-content;'>
                <span style='color: black;'><strong>{score_line}</strong></span>
                </div>
                """, unsafe_allow_html=True)
                
                reasoning_start = section.find("REASONING:") + 10
                reasoning = section[reasoning_start:].strip()
                st.markdown(f"**Reasoning:**\n{reasoning}")
                
                st.markdown("---")
            
            elif section.startswith("OVERALL"):
                st.markdown("### Overall Assessment")
                st.markdown(section)

        return response
    except Exception as e:
        st.error(f"Error generating analysis: {str(e)}")
        return None

# Sidebar setup
with st.sidebar:
    options = option_menu(
        menu_title="Navigation",
        options=["Home", "Self Evaluation"],
        icons=["house", "person-check"],
        menu_icon="person-check",
        default_index=0,
        styles={
            "container": {"padding": "0!important"},
            "icon": {"color": "#FFD700", "font-size": "20px"},
            "nav-link": {"font-size": "17px", "text-align": "left", "margin": "5px"},
            "nav-link-selected": {"background-color": "#2E2E2E"}
        }
    )
    
    # Add some spacing
    st.markdown("---")
    
    # API Key section
    st.markdown('<p style="color: white;">OpenAI API Key:</p>', unsafe_allow_html=True)
    col1, col2 = st.columns([5,1], gap="small")
    with col1:
        api_key = st.text_input('', type='password', label_visibility="collapsed")
    with col2:
        check_api = st.button('>', key='api_button')
    
    if check_api:
        if not api_key:
            st.warning('Please enter your OpenAI API token!')
        else:
            try:
                client = OpenAI(api_key=api_key)
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": "Hello"}],
                    max_tokens=5
                )
                st.session_state.api_key = api_key
                st.session_state.api_key_valid = True
                st.success('API key is valid!')
            except Exception as e:
                st.error('Invalid API key or API error occurred')
                st.session_state.api_key_valid = False

# Home page content update
if options == "Home":
    st.markdown("<h1 style='text-align: center; margin-bottom: 15px; color: white;'>Welcome to Self-Eval Assistant!</h1>", unsafe_allow_html=True)
    
    st.markdown("<div style='text-align: center; padding: 10px; margin-bottom: 20px; font-size: 18px; color: white;'>Self-Eval Assistant helps you analyze your work accomplishments and prepare comprehensive self-evaluations. Our AI-powered system helps identify key achievements, growth areas, and impact metrics.</div>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("<h3 style='text-align: center; color: #FFD700; margin-bottom: 10px;'>Key Features</h3>", unsafe_allow_html=True)
        st.markdown("""
        <div style='background-color: #f8f9fa; padding: 15px; border-radius: 10px; font-size: 16px; color: black; min-height: 200px;'>
        <ul style='list-style-type: none; padding-left: 0; margin: 0;'>
        <li style='margin-bottom: 8px;'>• Performance Analysis</li>
        <li style='margin-bottom: 8px;'>• Achievement Tracking</li>
        <li style='margin-bottom: 8px;'>• Impact Measurement</li>
        <li style='margin-bottom: 8px;'>• Growth Opportunity Identification</li>
        <li style='margin-bottom: 8px;'>• Skills Assessment</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

# Self Evaluation page remains the same
elif options == "Self Evaluation":
    st.title("Performance Analysis")
    
    uploaded_file = st.file_uploader("Upload your evaluation data (CSV/XLSX)", type=['csv', 'xlsx'])
    
    if uploaded_file is not None:
        try:
            # Check file extension and read accordingly
            file_extension = uploaded_file.name.split('.')[-1]
            
            if file_extension == 'csv':
                df = pd.read_csv(uploaded_file)
            elif file_extension == 'xlsx':
                df = pd.read_excel(uploaded_file)
            
            st.session_state.df = df
            
            # Add tabs for different evaluation parts
            tab1, tab2, tab3 = st.tabs(["Part 1: Goals", "Part 2: Enablement", "Part 3: Innovation"])
            
            with tab1:
                st.header("Strategic Objectives Evaluation")
                if st.button("Generate Goals Analysis", key="goals"):
                    analysis = generate_evaluation_analysis(df, 1)
                    if analysis:
                        st.markdown(analysis)
            
            with tab2:
                st.header("Behavioral Competencies Evaluation")
                if st.button("Generate Enablement Analysis", key="enablement"):
                    analysis = generate_evaluation_analysis(df, 2)
                    if analysis:
                        st.markdown(analysis)
            
            with tab3:
                st.header("Innovation Evaluation")
                if st.button("Generate Innovation Analysis", key="innovation"):
                    analysis = generate_evaluation_analysis(df, 3)
                    if analysis:
                        st.markdown(analysis)

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
