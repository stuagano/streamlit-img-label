# database_setup.py
import streamlit as st
import sqlite3
import pandas as pd

def init_db():
    conn = sqlite3.connect('pdf_data.db')
    c = conn.cursor()
    
    # Create table with appropriate columns
    c.execute('''
        CREATE TABLE IF NOT EXISTS extracted_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pdf_file TEXT,
            page_number INTEGER,
            label TEXT,
            text TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    return conn

def get_db_stats():
    conn = sqlite3.connect('pdf_data.db')
    c = conn.cursor()
    
    # Get total records
    c.execute("SELECT COUNT(*) FROM extracted_data")
    total_records = c.fetchone()[0]
    
    # Get unique PDFs
    c.execute("SELECT COUNT(DISTINCT pdf_file) FROM extracted_data")
    unique_pdfs = c.fetchone()[0]
    
    return total_records, unique_pdfs

def view_recent_entries(limit=10):
    conn = sqlite3.connect('pdf_data.db')
    query = f"SELECT * FROM extracted_data ORDER BY timestamp DESC LIMIT {limit}"
    return pd.read_sql_query(query, conn)
