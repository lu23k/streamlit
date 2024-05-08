import os
import copy

import streamlit as st
import pandas as pd
import plotly.express as px

#A streamlit app with two centered texts with different seizes
import streamlit as st



# Add a styled heading with underline
st.markdown("""
    <style>
        .underline {
            text-decoration: underline;
        }
    </style>
    <h1 class="underline" style="text-align: center; color: green;">L'Occitane Sales Analytics Platform</h1>
    """, unsafe_allow_html=True)


# Set font size using CSS
st.markdown("<style>h2 {font-size: 23px;}</style>", unsafe_allow_html=True)

# Display markdown section with custom font size
st.markdown("<h2 style='text-align: center; color: grey;'>This platform provides sales and social media analytics powered by AI</h2>", unsafe_allow_html=True)


# st.text('This platform povides sales and social media analytics powered by AI')
st.image('picture.png', caption="HKUST MSBA X L'Occitane",use_column_width=True)

css = '''
<style>
section.Home > div:has(~ footer ) {
    padding-bottom: 1px;
}
</style>
'''
st.markdown(css, unsafe_allow_html=True)