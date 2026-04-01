# content_based_filtering.py - Optimized for Maximum Performance
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import TruncatedSVD
import warnings
from scipy.sparse import hstack
warnings.filterwarnings('ignore')

class ContentBasedRecommender:
    def __init__(self, books_df, ratings_df):
        """
        Enhanced Content-Based Filtering with optimized feature engineering
        and improved similarity computation for maximum performance metrics.
        """
        self.books_df = books_df.copy()
        self.ratings_df = ratings_df.copy()
        
        # Enhanced similarity computation
        self.similarity_matrix = None
        self.tfidf_matrix = None
        self.feature_matrix = None
        self.book_id_to_idx = {}
        self.idx_to_book_id = {}
        
        # Performance optimization caches
        self._user_profiles_cache = {}
        self._book_features_cache = {}
        
        # Enhanced preprocessing and feature building
        self._preprocess_data_enhanced()
        self._build_content_features_enhanced()

        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.tfidf_matrix = self.vectorizer.fit_transform(self.books_df["combined_features"])
    
    def _preprocess_data_enhanced(self):
        """Enhanced data preprocessing with better feature extraction"""
        # Ensure required columns with intelligent defaults
        default_columns = {
            'ratings_count': 10.0,
            'average_rating': 3.5,
            'books_count': 5.0,
            'original_publication_year': 2000.0,
            'authors': 'Unknown Author',
            'original_title': '',
            'title': '',
            'language_code': 'eng',
        }
        
        for col, default_val in default_columns.items():
            if col not in self.books_df.columns:
                self.books_df[col] = default_val
            else:
                self.books_df[col] = self.books_df[col].fillna(default_val)

        # Enhanced text cleaning
        b = self.books_df.reset_index(drop=True)
        b['authors'] = b['authors'].fillna('Unknown Author').astype(str)
        b['original_title'] = b['original_title'].fillna('').astype(str)
        b['title'] = b['title'].fillna('').astype(str)
        b['language_code'] = b['language_code'].fillna('eng').astype(str)
        
        # Clean and normalize publication years
        b['original_publication_year'] = pd.to_numeric(b['original_publication_year'], errors='coerce')
        b['original_publication_year'] = b['original_publication_year'].fillna(b['original_publication_year'].median())
        
        self.books_df = b

        # Build optimized mapping
        self.book_id_to_idx = {int(b.loc[i, 'book_id']): i for i in range(len(b))}
        self.idx_to_book_id = {i: int(b.loc[i, 'book_id']) for i in range(len(b))}

        # Enhanced combined features with better text processing
        b['combined_features'] = (
            b['title'].str.lower() + ' ' +
            b['authors'].str.lower().str.replace(',', ' ') + ' ' +
            b['original_title'].str.lower() + ' ' +
            b['language_code'].str.lower()
        )
        
        # Add genre information if available (extract from title patterns)
        self._extract_genre_features(b)
    
    def _get_user_profile(self, user_id):
        """Backward-compatible alias used by the metrics analyzer."""
        return self._get_user_profile_enhanced(user_id)
    
    def _extract_genre_features(self, df):
        """Extract genre-like features from titles and other metadata"""
        # Common genre keywords to look for
        genre_keywords = {
            'fantasy': ['fantasy', 'magic', 'wizard', 'dragon', 'kingdom'],
            'romance': ['love', 'heart', 'romance', 'wedding', 'kiss'],
            'mystery': ['mystery', 'detective', 'murder', 'case', 'investigation'],
            'scifi': ['space', 'future', 'robot', 'alien', 'planet'],
            'historical': ['war', 'history', 'century', 'ancient', 'medieval'],
            'thriller': ['thriller', 'danger', 'chase', 'escape', 'terror'],
            'biography': ['life', 'biography', 'memoir', 'story of'],
            'young_adult': ['teen', 'young', 'school', 'coming of age']
        }
        
        # Extract genre features
        for genre, keywords in genre_keywords.items():
            pattern = '|'.join(keywords)
            df[f'genre_{genre}'] = df['combined_features'].str.contains(pattern, case=False, na=False).astype(int)
        
        # Add these to combined features
        genre_cols = [f'genre_{genre}' for genre in genre_keywords.keys()]
        genre_text = df[genre_cols].apply(lambda x: ' '.join([col.replace('genre_', '') for col, val in x.items() if val == 1]), axis=1)
        df['combined_features'] = df['combined_features'] + ' ' + genre_text
    
    
    def _compute_enhanced_similarity(self):
        """Compute enhanced similarity matrix with multiple similarity measures"""
        try:
            # Primary: Cosine similarity
            cosine_sim = cosine_similarity(self.feature_matrix)
            
            # Secondary: Pearson correlation for numerical features
            numerical_part = self.feature_matrix[:, -100:]  # Last 100 features are numerical
            pearson_sim = np.corrcoef(numerical_part)
            pearson_sim = np.nan_to_num(pearson_sim, nan=0.0)
            
            # Combine similarities with weights
            self.similarity_matrix = 0.8 * cosine_sim + 0.2 * pearson_sim
            
            # Apply non-linear transformation to enhance top similarities
            self.similarity_matrix = np.power(self.similarity_matrix, 1.2)
            
        except Exception as e:
            self.similarity_matrix = cosine_similarity(self.feature_matrix)
    
        
    def _build_content_features_enhanced(self):
        # Ensure combined text features exist
        if "combined_features" not in self.books_df.columns:
            self.books_df["combined_features"] = (
                self.books_df["title"].fillna("") + " " +
                self.books_df["author"].fillna("") + " " +
                self.books_df["publisher"].fillna("") + " " +
                self.books_df["genres"].fillna("")
            )
        
        # ---- TEXT FEATURES ----
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.tfidf_matrix = self.vectorizer.fit_transform(self.books_df["combined_features"])
        
        # ---- NUMERIC FEATURES ----
        # Fill missing values with 0 to avoid NaNs
        self.numeric_features = self.books_df[
            ["average_rating", "original_publication_year", "ratings_count"]
        ].fillna(0).values
        
        # Normalize numeric features (important for scale balance)
        self.numeric_features = (
            (self.numeric_features - np.mean(self.numeric_features, axis=0)) /
            (np.std(self.numeric_features, axis=0) + 1e-8)
        )
        
        # ---- FINAL FEATURE MATRIX ----
        self.feature_matrix = hstack([self.tfidf_matrix, self.numeric_features])
        
        # Build index mappings for book lookup
        self.book_id_to_idx = {book_id: idx for idx, book_id in enumerate(self.books_df["book_id"])}
        self.idx_to_book_id = {idx: book_id for book_id, idx in self.book_id_to_idx.items()}
        
    def _build_query_feature_vector(self, query_text):
        # Transform query into TF-IDF vector
        query_tfidf = self.vectorizer.transform([query_text])
        
        # For numeric features in query mode, use zeros (since no metadata is provided by user)
        query_numeric = np.zeros((1, self.numeric_features.shape[1]))
        print("1234")
        # Combine
        return hstack([query_tfidf, query_numeric])

        
    def recommend_from_query(self, query_text: str, n_recommendations: int = 10):
        try:
            if not query_text or not str(query_text).strip():
                return self._get_popular_books_enhanced(n_recommendations)

            candidates = {}

            # --- 1. Weighted substring match ---
            title_mask = self.books_df['title'].astype(str).str.contains(str(query_text), case=False, na=False)
            author_mask = self.books_df['authors'].astype(str).str.contains(str(query_text), case=False, na=False)
            year_mask = self.books_df['original_publication_year'].astype(str).str.contains(str(query_text), case=False, na=False)

            for idx, row in self.books_df.iterrows():
                score = 0.0
                if title_mask.iloc[idx]:
                    score += 3.0
                if author_mask.iloc[idx]:
                    score += 2.0
                if year_mask.iloc[idx]:
                    score += 1.0
                if score > 0:
                    # small popularity boost
                    score += np.log1p(row['ratings_count']) / 10.0
                    candidates[row['book_id']] = max(candidates.get(row['book_id'], 0), score)

            # --- 2. TF-IDF similarity ---
            q_vec = self._build_query_feature_vector(query_text)
            sims = cosine_similarity(q_vec, self.feature_matrix)[0]
            sims = np.maximum(0, sims)  
            sims = np.power(sims, 1.2)

            for idx, sim in enumerate(sims):
                if sim > 0:
                    book_id = self.idx_to_book_id[idx]
                    book_info = self.books_df.iloc[idx]
                    # add popularity & quality boosts
                    popularity_boost = min(1.2, 1.0 + np.log1p(book_info['ratings_count']) / 1000)
                    quality_boost = min(1.3, book_info['average_rating'] / 3.5 if book_info['average_rating'] > 0 else 1.0)
                    score = sim * popularity_boost * quality_boost
                    candidates[book_id] = candidates.get(book_id, 0) + score  # merge with substring score

            # --- Final ranking ---
            if not candidates:
                return self._get_popular_books_enhanced(n_recommendations)

            sorted_books = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
            top_ids = [bid for bid, _ in sorted_books[:n_recommendations]]

            result = self.books_df[self.books_df['book_id'].isin(top_ids)].copy()
            score_map = dict(sorted_books[:n_recommendations])
            result['query_score'] = result['book_id'].map(score_map)
            return result.sort_values('query_score', ascending=False)

        except Exception as e:
            # safe fallback
            print(f"[Content] recommend_from_query error: {e}")
            return self._get_popular_books_enhanced(n_recommendations)

    
    def _get_popular_books_enhanced(self, n_recommendations=10):
        """Enhanced fallback to popular books with quality filtering"""
        try:
            # Filter for quality books
            quality_books = self.books_df[
                (self.books_df['average_rating'] >= 3.8) & 
                (self.books_df['ratings_count'] >= 100)
            ].copy()
            
            if len(quality_books) < n_recommendations:
                quality_books = self.books_df.copy()
            
            # Score by combined popularity and quality
            quality_books['popularity_score'] = (
                np.log1p(quality_books['ratings_count']) * 0.6 +
                quality_books['average_rating'] * 0.4
            )
            
            return quality_books.nlargest(n_recommendations, 'popularity_score')
            
        except Exception as e:
            return self.books_df.head(n_recommendations)
    
    def get_book_similarity(self, book_id1, book_id2):
        """Enhanced book similarity computation"""
        if book_id1 in self.book_id_to_idx and book_id2 in self.book_id_to_idx:
            idx1 = self.book_id_to_idx[book_id1]
            idx2 = self.book_id_to_idx[book_id2]
            return float(self.similarity_matrix[idx1][idx2])
        return 0.0
    
    def get_similar_books(self, book_id, n_similar=5):
        """Enhanced similar books recommendation"""
        if book_id not in self.book_id_to_idx:
            return pd.DataFrame()
        
        book_idx = self.book_id_to_idx[book_id]
        similarities = self.similarity_matrix[book_idx]
        
        # Get top similar books (excluding the book itself)
        similar_indices = np.argsort(similarities)[::-1][1:n_similar+1]
        similar_book_ids = [self.idx_to_book_id[idx] for idx in similar_indices]
        
        similar_books = self.books_df[self.books_df['book_id'].isin(similar_book_ids)].copy()
        similarity_scores = [similarities[idx] for idx in similar_indices]
        similar_books['similarity'] = similarity_scores
        
        return similar_books.sort_values('similarity', ascending=False)