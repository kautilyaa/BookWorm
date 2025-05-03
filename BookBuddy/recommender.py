import pandas as pd
import numpy as np
import streamlit as st

def get_item_based_recommendations(book_isbn, item_similarity_df, books_df, top_n=5):
    """
    Generate item-based collaborative filtering recommendations with robust handling
    of edge cases and numerical issues.
    
    Args:
        book_isbn: ISBN of the book for which to find recommendations
        item_similarity_df: DataFrame containing item-item similarity matrix
        books_df: DataFrame containing book metadata
        top_n: Number of recommendations to return
        
    Returns:
        DataFrame containing recommended books
    """
    try:
        # Convert ISBN to string if it's not already
        book_isbn = str(book_isbn)
        
        # Check if the book ISBN exists in the similarity matrix
        if item_similarity_df is None or item_similarity_df.empty:
            st.warning("Similarity matrix is empty or not available.")
            # Fall back to popularity-based recommendations
            return get_popularity_based_recommendations(book_isbn, books_df, top_n)
            
        if book_isbn not in item_similarity_df.index:
            st.warning(f"Book with ISBN {book_isbn} not found in the similarity matrix.")
            # Fall back to popularity-based recommendations
            return get_popularity_based_recommendations(book_isbn, books_df, top_n)
        
        # Get similarity scores for the target book
        try:
            similar_books = item_similarity_df[book_isbn].sort_values(ascending=False)
            
            # Check if we have valid similarity scores
            if similar_books.empty or similar_books.isna().all():
                st.warning("No valid similarity scores found.")
                return get_popularity_based_recommendations(book_isbn, books_df, top_n)
            
            # Remove the book itself from recommendations
            similar_books = similar_books.drop(book_isbn, errors='ignore')
            
            # Remove any NaN or infinite values
            similar_books = similar_books[~np.isnan(similar_books) & ~np.isinf(similar_books)]
            
            # Get top N most similar books
            top_similar_books = similar_books.head(top_n)
            
            # If we have fewer than 2 recommendations, supplement with popular books
            if len(top_similar_books) < 2:
                popular_recs = get_popularity_based_recommendations(book_isbn, books_df, top_n)
                if not popular_recs.empty:
                    # Only add popular books that aren't already in our recommendations
                    existing_isbns = set(top_similar_books.index)
                    for _, row in popular_recs.iterrows():
                        if row['ISBN'] not in existing_isbns and len(top_similar_books) < top_n:
                            # Add this book to our recommendations with a lower similarity score
                            similar_isbn = row['ISBN']
                            top_similar_books[similar_isbn] = 0.1  # Low similarity score
            
            # Get book details for the recommended books
            recommended_books = books_df[books_df['ISBN'].isin(top_similar_books.index)]
            
            # If somehow we don't have metadata for some books, filter those out
            valid_isbns = set(recommended_books['ISBN'])
            top_similar_books = top_similar_books[top_similar_books.index.isin(valid_isbns)]
            
            if len(top_similar_books) == 0:
                return get_popularity_based_recommendations(book_isbn, books_df, top_n)
            
            # Sort by similarity score
            try:
                recommended_books = recommended_books.set_index('ISBN').loc[top_similar_books.index].reset_index()
            except:
                # Handle case where books might not be in the expected order
                recommended_books = recommended_books.set_index('ISBN')
                # Only keep the books that are in both dataframes
                common_isbns = list(set(recommended_books.index) & set(top_similar_books.index))
                if common_isbns:
                    recommended_books = recommended_books.loc[common_isbns].reset_index()
                    # Add similarity column for reference
                    recommended_books['Similarity'] = recommended_books['ISBN'].map(top_similar_books)
                    recommended_books = recommended_books.sort_values('Similarity', ascending=False)
                else:
                    recommended_books = pd.DataFrame()
            
            return recommended_books
            
        except Exception as e:
            st.error(f"Error getting recommendations: {str(e)}")
            return get_popularity_based_recommendations(book_isbn, books_df, top_n)
            
    except Exception as e:
        st.error(f"Unexpected error in recommendation engine: {str(e)}")
        # Fall back to popularity-based recommendations
        return get_popularity_based_recommendations(book_isbn, books_df, top_n)


def get_popularity_based_recommendations(book_isbn, books_df, top_n=5):
    """
    Fall back to popularity-based recommendations when collaborative filtering fails.
    
    Args:
        book_isbn: ISBN of the book for reference (to exclude from recommendations)
        books_df: DataFrame containing book metadata
        top_n: Number of recommendations to return
        
    Returns:
        DataFrame containing recommended books based on popularity/rating
    """
    try:
        # Check if the books dataframe is valid
        if books_df is None or books_df.empty:
            st.warning("No book data available for recommendations.")
            return pd.DataFrame()
        
        # Get the book's author to recommend other books by the same author
        book_row = books_df[books_df['ISBN'] == book_isbn]
        
        # If the book is found, get other books by the same author
        if not book_row.empty:
            author = book_row.iloc[0]['Book-Author']
            
            # Get other books by the same author
            author_books = books_df[
                (books_df['Book-Author'] == author) & 
                (books_df['ISBN'] != book_isbn)
            ]
            
            # If we have books by the same author, recommend those first
            if len(author_books) > 0:
                # Sort by rating if available
                if 'Average-Rating' in author_books.columns:
                    author_books = author_books.sort_values('Average-Rating', ascending=False)
                
                # Return top N books by the same author
                return author_books.head(top_n)
        
        # If no author matches or not enough books, return top rated books
        if 'Average-Rating' in books_df.columns:
            # Exclude the current book
            popular_books = books_df[books_df['ISBN'] != book_isbn]
            popular_books = popular_books.sort_values('Average-Rating', ascending=False)
            return popular_books.head(top_n)
        else:
            # If no rating available, just return some random books
            return books_df[books_df['ISBN'] != book_isbn].sample(min(top_n, len(books_df)-1))
            
    except Exception as e:
        st.error(f"Error getting popularity recommendations: {str(e)}")
        # Last resort - return empty DataFrame
        return pd.DataFrame()

def get_user_recommendations(user_id, ratings_df, item_similarity_df, books_df, top_n=5):
    """
    Generate personalized recommendations for a user based on their ratings,
    with robust error handling and fallback mechanisms.
    
    Args:
        user_id: ID of the user for whom to generate recommendations
        ratings_df: DataFrame containing user-item ratings
        item_similarity_df: DataFrame containing item-item similarity matrix
        books_df: DataFrame containing book metadata
        top_n: Number of recommendations to return
        
    Returns:
        DataFrame containing recommended books
    """
    try:
        # Verify input data
        if item_similarity_df is None or item_similarity_df.empty:
            st.warning("Similarity matrix is empty or not available.")
            return get_popular_books(books_df, top_n)
            
        if ratings_df is None or ratings_df.empty:
            st.warning("No ratings data available.")
            return get_popular_books(books_df, top_n)
        
        # Convert user_id to consistent type
        try:
            user_id = int(user_id)
        except:
            # If conversion fails, keep as is
            pass
        
        # Get books rated by the user
        user_ratings = ratings_df[ratings_df['User-ID'] == user_id]
        
        if user_ratings.empty:
            st.warning(f"No ratings found for user {user_id}")
            return get_popular_books(books_df, top_n)
        
        # Initialize a dictionary to store weighted ratings
        weighted_ratings = {}
        
        # Count how many books we processed successfully
        processed_books = 0
        
        # For each book the user has rated
        for _, row in user_ratings.iterrows():
            try:
                book_isbn = str(row['ISBN'])  # Ensure consistent type
                
                # Convert rating to float and validate
                try:
                    rating = float(row['Book-Rating'])
                except:
                    continue  # Skip invalid ratings
                
                # Skip if rating is 0 (user hasn't explicitly rated the book)
                if rating == 0:
                    continue
                
                # Skip if book is not in the similarity matrix
                if book_isbn not in item_similarity_df.index:
                    continue
                
                # Get similar books
                similar_books = item_similarity_df[book_isbn]
                
                # Skip if similar_books is invalid
                if similar_books is None or similar_books.empty:
                    continue
                
                # Filter out any NaN or infinite values
                similar_books = similar_books[~np.isnan(similar_books) & ~np.isinf(similar_books)]
                
                # For each similar book
                for similar_isbn, similarity in similar_books.items():
                    try:
                        # Skip invalid values
                        if pd.isna(similarity) or np.isinf(similarity):
                            continue
                            
                        # Skip the book itself
                        if similar_isbn == book_isbn:
                            continue
                            
                        # Skip books already rated by the user
                        if similar_isbn in user_ratings['ISBN'].values:
                            continue
                            
                        # Calculate weighted rating based on similarity and user's rating
                        if similar_isbn not in weighted_ratings:
                            weighted_ratings[similar_isbn] = 0
                            
                        weighted_ratings[similar_isbn] += similarity * rating
                    except:
                        # Skip any items that cause errors
                        continue
                
                processed_books += 1
                
            except Exception as e:
                # Skip any books that cause errors
                continue
        
        # If we couldn't process any books successfully, fall back to popular books
        if processed_books == 0:
            st.warning("Could not process any user ratings successfully.")
            return get_popular_books(books_df, top_n)
        
        # Convert to DataFrame and sort
        if weighted_ratings:
            try:
                recommendations = pd.DataFrame(list(weighted_ratings.items()), columns=['ISBN', 'weighted_rating'])
                # Handle potential errors with sorting
                if 'weighted_rating' in recommendations.columns:
                    recommendations = recommendations.sort_values('weighted_rating', ascending=False).head(top_n)
                else:
                    recommendations = recommendations.head(top_n)
                
                # Get book details for the recommended books
                if not recommendations.empty:
                    recommended_books = books_df[books_df['ISBN'].isin(recommendations['ISBN'])]
                    
                    # Sort by weighted rating
                    try:
                        recommended_books = pd.merge(recommended_books, recommendations, on='ISBN')
                        recommended_books = recommended_books.sort_values('weighted_rating', ascending=False)
                    except:
                        # If merge fails, still return the books we found
                        pass
                    
                    if not recommended_books.empty:
                        return recommended_books
            except Exception as e:
                st.error(f"Error processing recommendations: {str(e)}")
        
        # Fallback if no valid recommendations
        st.info("Falling back to popular books recommendations.")
        return get_popular_books(books_df, top_n)
        
    except Exception as e:
        st.error(f"Error in user recommendation system: {str(e)}")
        return get_popular_books(books_df, top_n)


def get_popular_books(books_df, top_n=5):
    """Get popular books based on average rating as a fallback."""
    try:
        if books_df is None or books_df.empty:
            return pd.DataFrame()
            
        # If we have ratings data, use it to sort
        if 'Average-Rating' in books_df.columns:
            # Sort by rating (high to low)
            popular_books = books_df.sort_values('Average-Rating', ascending=False)
        else:
            # No ratings available, just return a sample
            popular_books = books_df
            
        # Return top N books
        return popular_books.head(top_n)
    except:
        # Last resort, return whatever subset we can
        try:
            return books_df.head(top_n)
        except:
            return pd.DataFrame()
