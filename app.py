import streamlit as st

st.title("Document Annotation Tool")

st.write("""
## Welcome to the Document Annotation System

This tool helps you:
1. Annotate document fields
2. Extract text from annotations
3. Process multiple documents
4. Manage annotation collections

### Available Pages:
- **Annotate**: Create and manage document annotations
- **Collections**: View and manage saved annotation collections
- **Data Extraction**: Process annotated documents
- **Import Records**: Batch process documents using templates
""")

# Add some helpful statistics or status if needed
st.sidebar.write("### Quick Stats")
st.sidebar.write("Navigate to Annotate to begin labeling documents")


