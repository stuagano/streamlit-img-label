# sql_management.py
import streamlit as st
import sqlite3
import pandas as pd

def load_table_data(table_name):
    """Load all data from specified table"""
    conn = sqlite3.connect('pdf_data.db')
    return pd.read_sql(f"SELECT * FROM {table_name}", conn)

def execute_sql_query(query):
    """Execute custom SQL query and return results"""
    conn = sqlite3.connect('pdf_data.db')
    return pd.read_sql(query, conn)

def create_view(view_name, query):
    """Create a new SQL view"""
    conn = sqlite3.connect('pdf_data.db')
    c = conn.cursor()
    c.execute(f"CREATE VIEW IF NOT EXISTS {view_name} AS {query}")
    conn.commit()

def get_saved_views():
    """Get list of existing views"""
    conn = sqlite3.connect('pdf_data.db')
    c = conn.cursor()
    c.execute("SELECT name FROM sqlite_master WHERE type='view'")
    return [view[0] for view in c.fetchall()]

def get_all_tables():
    """
    Get list of all tables in database.
    
    Returns:
        list: Names of all tables in the database
    """
    conn = sqlite3.connect('pdf_data.db')
    c = conn.cursor()
    # Query sqlite_master for table names
    c.execute("SELECT name FROM sqlite_master WHERE type='table'")
    # Convert tuple results to list of table names
    tables = [table[0] for table in c.fetchall()]
    conn.close()
    return tables

def get_unique_labels(table_name):
    """Get unique values from all columns in the selected table"""
    conn = sqlite3.connect('pdf_data.db')
    # Get column names first
    df = pd.read_sql(f"SELECT * FROM {table_name} LIMIT 1", conn)
    columns = df.columns.tolist()
    
    # Get unique values from each column
    unique_values = {}
    for col in columns:
        values = pd.read_sql(f"SELECT DISTINCT {col} FROM {table_name}", conn)
        unique_values[col] = values[col].tolist()
    return columns, unique_values

# SQL_Management.py (Streamlit page)
st.title("SQL Data Management")

# Tab-based interface
tab1, tab2, tab3 = st.tabs(["Table Viewer", "SQL Editor", "Saved Views"])

with tab1:
    st.header("Table Data Viewer")
    tables = get_all_tables()
    selected_table = st.selectbox("Select table to view:", tables)
    if selected_table:
        df = load_table_data(selected_table)
        st.dataframe(df, use_container_width=True)
        
        # Export functionality
        if st.button("Export to CSV"):
            df.to_csv(f"{selected_table}_export.csv", index=False)
            st.success("Data exported successfully!")

with tab2:
    st.header("SQL Query Editor")
    
    # Query Builder Section
    st.subheader("Query Builder")
    
    # Let user select table first
    tables = get_all_tables()
    selected_table = st.selectbox("Select table:", tables)
    
    if selected_table:
        # Get columns and their unique values
        columns, unique_values = get_unique_labels(selected_table)
        
        # Select column to filter on
        selected_column = st.selectbox("Select column to filter:", columns)
        
        if selected_column:
            # Select values from chosen column
            selected_values = st.multiselect(
                f"Select values from {selected_column}:", 
                unique_values[selected_column]
            )
            
            if selected_values:
                # Build query based on selection
                values_clause = ", ".join([f"'{value}'" for value in selected_values])
                built_query = f"SELECT * FROM {selected_table} WHERE {selected_column} IN ({values_clause})"
                
                # Show the generated query
                st.code(built_query, language='sql')
                
                # Preview results
                if st.button("Preview Query Results"):
                    results = execute_sql_query(built_query)
                    st.dataframe(results, use_container_width=True)
                
                # Save as view section
                col1, col2 = st.columns(2)
                with col1:
                    view_name = st.text_input("Enter view name:")
                with col2:
                    if st.button("Save Selection as View"):
                        if view_name:
                            try:
                                create_view(view_name, built_query)
                                st.success(f"View '{view_name}' created successfully!")
                            except Exception as e:
                                st.error(f"Error creating view: {str(e)}")
                        else:
                            st.warning("Please enter a view name")

with tab3:
    st.header("Saved Views")
    views = get_saved_views()
    selected_view = st.selectbox("Select view:", views)
    
    if selected_view:
        # View data
        df = execute_sql_query(f"SELECT * FROM {selected_view}")
        st.dataframe(df)
        
        # View definition
        conn = sqlite3.connect('pdf_data.db')
        c = conn.cursor()
        c.execute(f"SELECT sql FROM sqlite_master WHERE type='view' AND name='{selected_view}'")
        view_def = c.fetchone()[0]
        st.code(view_def, language='sql')
