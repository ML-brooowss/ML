import streamlit as st
from data_loader import load_books_data, load_recommendations, load_interactions_data
import pandas as pd

st.set_page_config(layout="wide")

# Load data outside of conditional blocks
df_books = load_books_data()
interactions = load_interactions_data()
recommendations_dict = load_recommendations()

df_books["i"] = df_books["i"].astype(str)
interactions["i"] = interactions["i"].astype(str)

# Initialize session states for show_recommendations and checkout_done
if 'show_recommendations' not in st.session_state:
    st.session_state.show_recommendations = False
if 'checkout_done' not in st.session_state:
    st.session_state.checkout_done = False
if 'borrowed_books' not in st.session_state:
    st.session_state.borrowed_books = []

# --- AUTHENTICATION STEP ---
if 'customer_authenticated' not in st.session_state:
    st.session_state.customer_authenticated = False
    st.session_state.selected_customer = None

if not st.session_state.customer_authenticated:
    st.title("Welcome to the Book Recommender System")
    st.subheader("Please enter your customer number to continue:")

    customer_list = interactions["u"].unique().tolist()

    with st.form("login_form"):
        selected_customer = st.selectbox("Select your customer number", sorted(customer_list))
        login_button = st.form_submit_button("Login")

        if login_button:
            st.session_state.selected_customer = selected_customer
            st.session_state.customer_authenticated = True
            st.rerun()  # force rerun immediately

else:
    # Logout form
    with st.form("logout_form"):
        st.write(f"Logged in as customer: **{st.session_state.selected_customer}**")
        logout_button = st.form_submit_button("Logout")

        if logout_button:
            st.session_state.customer_authenticated = False
            st.session_state.selected_customer = None
            st.session_state.borrowed_books = []
            st.rerun()()  # force rerun immediately

    selected_customer = st.session_state.selected_customer

    tab1, tab2, tab3 = st.tabs(["📚 Recommender", "🛒 Borrow Basket","📊 About & EDA"])

    with tab1:
        st.title("📚 Book Recommender System")
        st.header("Welcome to the recommender system of the Cantonal Library of Vaud")
        st.subheader("Project by Amélie Madrona & Linne Verhoeven")

        st.markdown(f"Logged in as customer: **{selected_customer}**")

        shuffle_clicked = st.button("🔀 Shuffle Previously Read Books")

        if 'recent_reads' not in st.session_state or shuffle_clicked:
            st.session_state.recent_reads = (
                interactions[interactions["u"] == selected_customer]
                .merge(df_books, on="i", how="left")
                .sample(5, replace=False)
            )

        st.subheader("Previously borrowed books 📚")

        read_books = st.session_state.recent_reads["title_clean"].dropna().tolist()

        if not read_books:
            st.info("No previously read books found for this customer.")
        else:
            cols = st.columns(5)
            for i, title in enumerate(read_books):
                book = df_books[df_books["title_clean"] == title]
                if not book.empty:
                    book = book.iloc[0]
                    with cols[i]:
                        st.markdown(f"""
                            <div style='height: 70px; overflow: hidden; text-align: center; font-weight: bold;'>
                                {title}
                            </div>
                        """, unsafe_allow_html=True)

                        img_url = book["image"]
                        image_height = 500

                        if pd.notnull(img_url):
                            st.markdown(f"""
                                <div style="height:{image_height}px; display:flex; align-items:center; justify-content:center; margin-bottom:8px;">
                                    <img src="{img_url}" style="max-height:{image_height}px; object-fit:contain;" />
                                </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                                <div style='height:{image_height}px; display:flex; align-items:center; justify-content:center;
                                            text-align:center; border:1px solid #ccc; border-radius:8px;
                                            background-color:#f9f9f9; padding:10px; color:black; margin-bottom:8px;'>
                                    No image available
                                </div>
                            """, unsafe_allow_html=True)

                        with st.expander("📖 More Info"):
                            st.markdown(f"""
                            **Author**: {book['author_clean'] if pd.notnull(book['author_clean']) else 'Unknown'}

                            **Published**: {book['PublishedDate'] if pd.notnull(book['PublishedDate']) else 'N/A'}  

                            **Language**: {book['Language'] if pd.notnull(book['Language']) else 'N/A'}
                            
                            **Description**:  
                            {book['Description'] if pd.notnull(book['Description']) else 'No description available for this book.'}

                            **Subjects**: {book['Subjects'] if pd.notnull(book['Subjects']) else 'N/A'}
    
                            **Link**: {"[🔗 View on Google Books](" + book['CanonicalLink'] + ")" if pd.notnull(book.get('CanonicalLink')) else '_Not available_'}
                            """)

                        # Borrow Again button BELOW the expander
                        if book['i'] not in st.session_state.borrowed_books:
                            if st.button(f"📚 Borrow Again: {title}", key=f"borrow_again_{book['i']}"):
                                st.session_state.borrowed_books.append(book['i'])
                                st.success(f"Added '{title}' to your borrow basket!")

        # Show Recommendations button only AFTER login
        if st.button("Show Recommendations"):
            st.session_state.show_recommendations = True

        if st.session_state.show_recommendations:
            st.subheader("Recommended books 📖")
            recommended_ids = recommendations_dict.get(selected_customer, "").split(" ")
            recommended_books = df_books[df_books["i"].isin(recommended_ids)]

            if recommended_books.empty:
                st.warning("No recommendations available for this customer.")
            else:
                for row in range(0, len(recommended_books), 5):
                    cols = st.columns(5)
                    for i, (_, book) in enumerate(recommended_books.iloc[row:row+5].iterrows()):
                        with cols[i]:
                            st.markdown(f"""
                                <div style='height: 70px; overflow: hidden; text-align: center; font-weight: bold;'>
                                    {book['title_clean']}
                                </div>
                            """, unsafe_allow_html=True)

                            img_url = book["image"]
                            image_height = 500

                            if pd.notnull(img_url):
                                st.markdown(f"""
                                    <div style="height:{image_height}px; display:flex; align-items:center; justify-content:center; margin-bottom:8px;">
                                        <img src="{img_url}" style="max-height:{image_height}px; object-fit:contain;" />
                                    </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.markdown(f"""
                                    <div style='height:{image_height}px; display:flex; align-items:center; justify-content:center;
                                                text-align:center; border:1px solid #ccc; border-radius:8px;
                                                background-color:#f9f9f9; padding:10px; color:black; margin-bottom:8px;'>
                                        No image available
                                    </div>
                                """, unsafe_allow_html=True)

                            with st.expander("📖 More Info"):
                                st.markdown(f"""
                                **Author**: {book['author_clean'] if pd.notnull(book['author_clean']) else 'Unknown'}

                                **Published**: {book['PublishedDate'] if pd.notnull(book['PublishedDate']) else 'N/A'}  
                                
                                **Language**: {book['Language'] if pd.notnull(book['Language']) else 'N/A'}

                                **Description**:  
                                {book['Description'] if pd.notnull(book['Description']) else 'No description available for this book.'}

                                **Subjects**: {book['Subjects'] if pd.notnull(book['Subjects']) else 'N/A'}
    
                                **Link**: {"[🔗 View on Google Books](" + book['CanonicalLink'] + ")" if pd.notnull(book.get('CanonicalLink')) else '_Not available_'}
                                """)

                            # Borrow button BELOW the expander
                            if book['i'] not in st.session_state.borrowed_books:
                                if st.button(f"📚 Borrow: {book['title_clean']}", key=f"borrow_{book['i']}"):
                                    st.session_state.borrowed_books.append(book['i'])
                                    st.success(f"Added '{book['title_clean']}' to your borrow basket!")

    with tab2:
        st.title("🛒 Borrow Basket")
        if st.session_state.borrowed_books:
            borrowed_books_df = df_books[df_books["i"].isin(st.session_state.borrowed_books)]

            for _, book in borrowed_books_df.iterrows():
                st.markdown(f"### {book['title_clean']}")
                img_url = book["image"]
                image_height = 100

                if pd.notnull(img_url):
                    st.image(img_url, use_container_width=False, width=150)
                else:
                    st.write("No image available")

                st.markdown(f"**Author**: {book['author_clean'] if pd.notnull(book['author_clean']) else 'Unknown'}")
                st.markdown(f"**Published**: {book['PublishedDate'] if pd.notnull(book['PublishedDate']) else 'N/A'}")
                st.markdown("---")

            if st.button("✅ Checkout"):
                st.session_state.borrowed_books.clear()
                st.balloons()
                st.session_state.checkout_done = True
                st.rerun()

        else:
            st.info("Your borrow basket is empty.")

        if st.session_state.checkout_done:
            st.success("Thank you for borrowing! Your basket is now empty.")
            st.balloons()
            # Reset flag so message shows only once
            st.session_state.checkout_done = False

    with tab3:
        st.title("📊 About This Recommender")
        st.header("Exploratory Data Analysis")
        st.markdown("""
            - Dataset size: ...
            - Most read genres: ...
            - Average ratings: ...
            - User engagement patterns: ...
        """)

        st.write("This plot shows genre popularity among users...")

        st.header("How Recommendations Work")
        st.markdown("""
            We used a collaborative filtering model that...
        """)