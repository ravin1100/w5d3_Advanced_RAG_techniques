import streamlit as st
import requests
import json
from typing import Dict, List
import plotly.graph_objects as go

# Configure the page
st.set_page_config(
    page_title="Sports Analytics RAG",
    page_icon="🏈",
    layout="wide"
)

# Constants
API_URL = "http://localhost:8000"

def create_processing_flow_diagram(steps: Dict[str, str]):
    """Create a visualization of the RAG processing steps"""
    fig = go.Figure()
    
    # Add nodes for each processing step
    y_pos = 0
    for step, details in steps.items():
        fig.add_trace(go.Scatter(
            x=[0],
            y=[y_pos],
            mode='markers+text',
            name=step,
            text=[step],
            marker=dict(size=30, symbol='circle'),
            textposition='right'
        ))
        y_pos -= 1
    
    # Update layout
    fig.update_layout(
        showlegend=False,
        height=400,
        title="RAG Processing Flow",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )
    
    return fig

def display_citations(citations: List[Dict[str, str]]):
    """Display citations in a formatted way"""
    for citation in citations:
        st.markdown(f"""
        **Source**: {citation.get('source', 'Unknown')}
        > {citation.get('text', '')}
        """)

def main():
    # Header
    st.title("🏈 Sports Analytics RAG System")
    st.markdown("""
    This system uses advanced RAG (Retrieval Augmented Generation) techniques to answer complex sports-related queries.
    It breaks down complex questions, searches through sports documents, and provides detailed answers with citations.
    """)
    
    # Input section
    st.header("📝 Enter Your Query")
    query = st.text_area(
        "Ask a complex sports question:",
        height=100,
        placeholder="Example: Which team has the best defense and how does their goalkeeper compare to the league average?"
    )
    
    # Sample queries
    st.markdown("### 📋 Sample Queries")
    sample_queries = [
        "What are the top 3 teams in defense and their key defensive statistics?",
        "Compare Messi's goal-scoring rate in the last season vs previous seasons",
        "Which goalkeeper has the best save percentage in high-pressure situations?"
    ]
    
    for sample in sample_queries:
        if st.button(f"Try: {sample[:50]}..."):
            query = sample
            st.session_state.query = sample
    
    # Process query
    if query and st.button("Process Query"):
        with st.spinner("Processing your query..."):
            try:
                # Call the backend API
                response = requests.post(
                    f"{API_URL}/process_query",
                    json={"query": query}
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Display processing flow
                    st.header("🔄 Processing Flow")
                    flow_fig = create_processing_flow_diagram(result["processing_steps"])
                    st.plotly_chart(flow_fig, use_container_width=True)
                    
                    # Display results in tabs
                    st.header("📊 Results")
                    tabs = st.tabs(["Main Answer"] + [f"Sub-Question {i+1}" for i in range(len(result["sub_questions"]))])
                    
                    # Main answer tab
                    with tabs[0]:
                        st.markdown("### Original Query")
                        st.info(result["original_query"])
                        
                        st.markdown("### Complete Answer")
                        for sub_q in result["sub_questions"]:
                            st.markdown(f"**{sub_q['sub_question']}**")
                            st.write(sub_q["answer"])
                            
                            st.markdown("#### Citations")
                            display_citations(sub_q["citations"])
                    
                    # Sub-question tabs
                    for i, sub_q in enumerate(result["sub_questions"], 1):
                        with tabs[i]:
                            st.markdown(f"### Sub-Question {i}")
                            st.info(sub_q["sub_question"])
                            
                            st.markdown("### Answer")
                            st.write(sub_q["answer"])
                            
                            st.markdown("### Citations")
                            display_citations(sub_q["citations"])
                            
                            # Display processing metrics
                            st.markdown("### Processing Details")
                            metrics = result["processing_steps"].get(f"sub_question_{i}", {})
                            for key, value in metrics.items():
                                st.metric(key.replace("_", " ").title(), value)
                
                else:
                    st.error(f"Error: {response.status_code} - {response.text}")
            
            except Exception as e:
                st.error(f"Error connecting to the backend: {str(e)}")

if __name__ == "__main__":
    main() 