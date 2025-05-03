import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

def display_rating_distribution(ratings_df):
    """
    Create a visualization for the distribution of book ratings.
    
    Args:
        ratings_df: DataFrame containing user ratings
        
    Returns:
        Plotly figure object
    """
    # Count the frequency of each rating
    rating_counts = ratings_df['Book-Rating'].value_counts().sort_index()
    
    # Create a bar chart
    fig = px.bar(
        x=rating_counts.index,
        y=rating_counts.values,
        labels={'x': 'Rating', 'y': 'Number of Ratings'},
        title='Distribution of Book Ratings',
        color=rating_counts.values,
        color_continuous_scale='Viridis'
    )
    
    # Customize layout
    fig.update_layout(
        xaxis_title='Rating (0-10)',
        yaxis_title='Number of Ratings',
        coloraxis_showscale=False
    )
    
    return fig

def display_top_authors(books_df, top_n=10):
    """
    Create a visualization for the top authors based on number of books.
    
    Args:
        books_df: DataFrame containing book metadata
        top_n: Number of authors to display
        
    Returns:
        Plotly figure object
    """
    # Count the number of books by each author
    author_counts = books_df['Book-Author'].value_counts().head(top_n)
    
    # Create a horizontal bar chart
    fig = px.bar(
        x=author_counts.values,
        y=author_counts.index,
        orientation='h',
        labels={'x': 'Number of Books', 'y': 'Author'},
        title=f'Top {top_n} Authors by Number of Books',
        color=author_counts.values,
        color_continuous_scale='Viridis'
    )
    
    # Customize layout
    fig.update_layout(
        xaxis_title='Number of Books',
        yaxis_title='Author',
        coloraxis_showscale=False,
        yaxis={'categoryorder': 'total ascending'}
    )
    
    return fig

def display_publication_year_trend(books_df):
    """
    Create a visualization for the trend of book publications over years.
    
    Args:
        books_df: DataFrame containing book metadata
        
    Returns:
        Plotly figure object
    """
    # Filter out books with invalid publication years
    valid_years = books_df[(books_df['Year-Of-Publication'] > 1800) & 
                           (books_df['Year-Of-Publication'] <= 2023)]
    
    # Group by publication year and count
    year_counts = valid_years.groupby('Year-Of-Publication').size().reset_index(name='count')
    
    # Create a line chart
    fig = px.line(
        year_counts,
        x='Year-Of-Publication',
        y='count',
        labels={'Year-Of-Publication': 'Publication Year', 'count': 'Number of Books'},
        title='Trend of Book Publications Over Years',
        markers=True
    )
    
    # Customize layout
    fig.update_layout(
        xaxis_title='Publication Year',
        yaxis_title='Number of Books'
    )
    
    return fig

def display_publisher_distribution(books_df, top_n=10):
    """
    Create a visualization for the distribution of books by publisher.
    
    Args:
        books_df: DataFrame containing book metadata
        top_n: Number of publishers to display
        
    Returns:
        Plotly figure object
    """
    # Count the number of books by each publisher
    publisher_counts = books_df['Publisher'].value_counts().head(top_n)
    
    # Create a pie chart
    fig = px.pie(
        values=publisher_counts.values,
        names=publisher_counts.index,
        title=f'Top {top_n} Publishers',
        hole=0.4
    )
    
    # Customize layout
    fig.update_layout(
        legend_title='Publisher'
    )
    
    return fig

def display_rating_heatmap(ratings_df, books_df, top_users=10, top_books=10):
    """
    Create a heatmap visualization for user-book ratings.
    
    Args:
        ratings_df: DataFrame containing user ratings
        books_df: DataFrame containing book metadata
        top_users: Number of top users to include
        top_books: Number of top books to include
        
    Returns:
        Plotly figure object
    """
    # Get top users by number of ratings
    top_user_ids = ratings_df['User-ID'].value_counts().head(top_users).index
    
    # Get top books by number of ratings
    top_book_isbns = ratings_df['ISBN'].value_counts().head(top_books).index
    
    # Filter ratings to include only top users and top books
    filtered_ratings = ratings_df[
        (ratings_df['User-ID'].isin(top_user_ids)) & 
        (ratings_df['ISBN'].isin(top_book_isbns))
    ]
    
    # Create a pivot table of user-book ratings
    pivot_ratings = filtered_ratings.pivot_table(
        index='User-ID',
        columns='ISBN',
        values='Book-Rating',
        fill_value=0
    )
    
    # Get book titles for the ISBNs
    books_map = books_df.set_index('ISBN')['Book-Title'].to_dict()
    
    # Replace ISBN column names with book titles
    column_titles = [books_map.get(isbn, isbn) for isbn in pivot_ratings.columns]
    
    # Create a heatmap
    fig = px.imshow(
        pivot_ratings.values,
        x=column_titles,
        y=pivot_ratings.index,
        labels=dict(x="Book", y="User ID", color="Rating"),
        title=f'Rating Heatmap (Top {top_users} Users, Top {top_books} Books)',
        color_continuous_scale='Viridis'
    )
    
    # Customize layout
    fig.update_layout(
        xaxis={'tickangle': 45},
        coloraxis_colorbar=dict(
            title='Rating',
            tickvals=[0, 5, 10],
            ticktext=['0', '5', '10']
        )
    )
    
    return fig
