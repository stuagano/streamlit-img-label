# Database_Management.py (New Streamlit page)
import streamlit as st
 


# database_setup.py
import streamlit as st
import sqlite3
import pandas as pd

def init_db():
    conn = sqlite3.connect('pdf_data.db')
    c = conn.cursor()
    
    c.execute("""
        CREATE TABLE IF NOT EXISTS extracted_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            page_number INTEGER NOT NULL,
            label TEXT NOT NULL,
            text TEXT,
            experience_type TEXT,
            experience_level TEXT, 
            owner TEXT,
            value_extracted TEXT,
            confidence_score FLOAT,
            extraction_status TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    return conn

def get_all_tables():
    """Get list of all tables in database"""
    conn = sqlite3.connect('pdf_data.db')
    c = conn.cursor()
    c.execute("SELECT name FROM sqlite_master WHERE type='table'")
    return [table[0] for table in c.fetchall()]

def delete_table(table_name):
    """Delete specified table from database"""
    conn = sqlite3.connect('pdf_data.db')
    c = conn.cursor()
    c.execute(f"DROP TABLE IF EXISTS {table_name}")
    conn.commit()

# In your Streamlit Database Management page:
st.title("Database Table Management")

# Display existing tables
tables = get_all_tables()
st.write("Existing Tables:", tables)

# Table deletion interface
if tables:
    table_to_delete = st.selectbox("Select table to delete:", tables)
    if st.button(f"Delete {table_to_delete}"):
        delete_table(table_to_delete)
        st.success(f"Table {table_to_delete} deleted successfully!")
        st.rerun()
def get_db_stats():
    conn = sqlite3.connect('pdf_data.db')
    c = conn.cursor()
    
    # Get total records
    c.execute("SELECT COUNT(*) FROM extracted_data")
    total_records = c.fetchone()[0]
    
    # Get unique PDFs
    c.execute("SELECT COUNT(DISTINCT filename) FROM extracted_data")
    unique_pdfs = c.fetchone()[0]
    
    return total_records, unique_pdfs

def view_recent_entries(limit=10):
    conn = sqlite3.connect('pdf_data.db')
    query = f"SELECT * FROM extracted_data ORDER BY timestamp DESC LIMIT {limit}"
    return pd.read_sql_query(query, conn)

st.title("Database Management")

if st.button("Initialize/Reset Database"):
    conn = init_db()
    st.success("Database initialized successfully!")

# Display database stats
total_records, unique_pdfs = get_db_stats()
st.metric("Total Records", total_records)
st.metric("Unique PDFs Processed", unique_pdfs)

# View recent entries
st.subheader("Recent Entries")
recent_df = view_recent_entries()
st.dataframe(recent_df)
