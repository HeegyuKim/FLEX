import streamlit as st
import jsonlines
import json
import io
import sqlparse
import glob



db_list = list(glob.glob("../bird_dev/llm/data/dev_databases/*/*.sqlite"))
db_dict = {db.split("/")[-1].replace(".sqlite", ""): db for db in db_list}


def format_sql(sql_query):
    formatted_sql = sqlparse.format(
        sql_query,
        reindent=True,
        keyword_case='upper',
        indent_width=4,
        strip_comments=False,
        wrap_after=80
    )
    return formatted_sql

if "row_index" not in st.session_state:
    st.session_state.row_index = 0

def load_jsonl(file):
    data = []
    string_data = io.StringIO(file.getvalue().decode("utf-8"))
    reader = jsonlines.Reader(string_data)
    for obj in reader:
        data.append(obj)
    return data

def show_row_content(data, row_index):
    if 0 <= row_index < len(data):
        st.subheader(f"Row {row_index} 내용:")
        for key, value in data[row_index].items():
            st.subheader(key)
            if "judge_result_category" == key:
                st.json(value)
                continue
            
            value = value if isinstance(value, str) else str(value)
            if not value:
                continue
            if "schema" in key:
                with st.expander("스키마 내용 보기"):
                    st.code(value, language="sql")
            elif "query" in key or "sql" in key:
                value = format_sql(value)
                st.code(value, language="sql")
            
            else:
                st.markdown(value.replace('\n', "    \n"))
    else:
        st.error("유효하지 않은 row index입니다.")

def execute_sql(query, db_path):
    if not query:
        st.error("SQL Query를 입력하세요.")
        return

    try:
        import sqlite3
        # read-only mode로 DB 연결
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        cursor = conn.cursor()
        cursor.execute(query)
        result = cursor.fetchall()
        conn.close()
    except Exception as e:
        st.error(f"SQL Query 실행 중 오류가 발생했습니다: {e}")
        return

    if not result:
        st.warning("결과가 없습니다.")
        return

    st.write(f"총 {len(result)}개의 row가 있습니다.")
    if len(result) > 100:
        st.warning("100개 이상의 row가 있습니다. 처음 100개의 row만 표시합니다.")
        result = result[:100]
        
    for row in result:
        st.write(row)

def main():
    st.title("JSONL 파일 뷰어")

    uploaded_file = st.file_uploader("JSONL 파일을 업로드하세요", type="jsonl")

    with st.sidebar:
        query_db = st.selectbox("DB 선택", db_dict.keys())
        query = st.text_area("SQL Query 입력", height=200)
        if st.button("Query 실행"):
            execute_sql(query, db_dict[query_db])


    if uploaded_file is not None:
        data = load_jsonl(uploaded_file)
        st.write(f"총 {len(data)}개의 row가 있습니다.")

        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Previous"):
                st.session_state.row_index = max(0, st.session_state.row_index - 1)
                st.rerun()
        
        with col2:
            st.session_state.row_index = st.number_input("보고 싶은 row의 index를 입력하세요", 
                                                         min_value=0, 
                                                         max_value=len(data)-1, 
                                                         value=st.session_state.row_index)
        
        with col3:
            if st.button("Next"):
                st.session_state.row_index = min(len(data) - 1, st.session_state.row_index + 1)
                st.rerun()

        show_row_content(data, st.session_state.row_index)

if __name__ == "__main__":
    main()