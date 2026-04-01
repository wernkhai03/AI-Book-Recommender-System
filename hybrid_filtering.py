import pandas as pd
import numpy as np
from typing import Optional, Dict, TYPE_CHECKING
import warnings
warnings.filterwarnings('ignore')

if TYPE_CHECKING:
    from content_based_filtering import ContentBasedRecommender
    from collaborative_filtering import CollaborativeFilteringRecommender

class HybridRecommender:
    """
    Enhanced Hybrid Recommender with optimized performance and advanced ensemble methods.
    Features:
    - Intelligent adaptive weighting based on user characteristics
    - Advanced diversity optimization
    - Performance-optimized scoring with caching
    - Enhanced explanation generation
    """

    def __init__(
        self,
        books_df: pd.DataFrame,
        ratings_df: pd.DataFrame,
        content_recommender: Optional["ContentBasedRecommender"] = None,
        collaborative_recommender: Optional["CollaborativeFilteringRecommender"] = None
    ):
        self.books_df = books_df.copy()
        self.ratings_df = ratings_df.copy()
        
        # Use provided recommenders or create new ones
        if content_recommender:
            self.content_recommender = content_recommender
        else:
            from content_based_filtering import ContentBasedRecommender
            self.content_recommender = ContentBasedRecommender(self.books_df, self.ratings_df)
            
        if collaborative_recommender:
            self.collaborative_recommender = collaborative_recommender
        else:
            from collaborative_filtering import CollaborativeFilteringRecommender
            self.collaborative_recommender = CollaborativeFilteringRecommender(self.ratings_df, self.books_df)

        # Enhanced user analysis
        self._build_user_profiles()
        self.book_rows = self.books_df.set_index("book_id", drop=False)
        
        # Performance caches
        self._recommendation_cache = {}
        self._user_weight_cache = {}

    def _build_user_profiles(self):
        """Build comprehensive user profiles for better recommendations"""
        self.user_to_items = {}
        self.user_characteristics = {}
        
        for _, row in self.ratings_df.iterrows():
            try:
                user_id = int(row.iloc[0])
                book_id = int(row.iloc[1])
                rating = float(row.iloc[2])
                
                if user_id not in self.user_to_items:
                    self.user_to_items[user_id] = {}
                self.user_to_items[user_id][book_id] = rating
            except Exception:
                continue
        
        # Compute user characteristics
        for user_id, ratings_dict in self.user_to_items.items():
            ratings_list = list(ratings_dict.values())
            self.user_characteristics[user_id] = {
                'rating_count': len(ratings_list),
                'avg_rating': np.mean(ratings_list),
                'rating_variance': np.var(ratings_list),
                'rating_range': max(ratings_list) - min(ratings_list),
                'high_ratings_ratio': sum(1 for r in ratings_list if r >= 4) / len(ratings_list)
            }

    def _adaptive_weights_enhanced(self, user_id: int):
        """Enhanced adaptive weighting based on comprehensive user analysis"""
        if user_id in self._user_weight_cache:
            return self._user_weight_cache[user_id]
        
        user_chars = self.user_characteristics.get(user_id, {})
        rating_count = user_chars.get('rating_count', 0)
        rating_variance = user_chars.get('rating_variance', 1.0)
        high_ratings_ratio = user_chars.get('high_ratings_ratio', 0.5)
        
        # Base weights
        content_weight = 0.4
        collaborative_weight = 0.6
        
        # Adjust based on user activity
        if rating_count < 10:
            # New users: prefer content-based
            content_weight = 0.7
            collaborative_weight = 0.3
        elif rating_count < 25:
            # Moderate users: balanced approach
            content_weight = 0.5
            collaborative_weight = 0.5
        else:
            # Active users: prefer collaborative
            content_weight = 0.3
            collaborative_weight = 0.7
        
        # Adjust based on rating behavior
        if rating_variance > 2.0:
            # High variance users benefit more from content-based
            content_weight += 0.1
            collaborative_weight -= 0.1
        
        if high_ratings_ratio > 0.8:
            # Users who rate everything highly benefit from collaborative
            collaborative_weight += 0.1
            content_weight -= 0.1
        
        # Ensure weights sum to 1
        total_weight = content_weight + collaborative_weight
        content_weight /= total_weight
        collaborative_weight /= total_weight
        
        weights = (content_weight, collaborative_weight)
        self._user_weight_cache[user_id] = weights
        
        return weights

    def _get_recommendations_from_algorithms(self, user_id: int, n_recommendations: int):
        """Get recommendations from both algorithms with error handling"""
        content_recs = pd.DataFrame()
        collaborative_recs = pd.DataFrame()
        
        try:
            content_recs = self.content_recommender.recommend(user_id, n_recommendations * 3)
        except Exception as e:
            print(f"Content-based recommendation failed: {e}")
        
        try:
            collaborative_recs = self.collaborative_recommender.recommend(user_id, n_recommendations * 3)
        except Exception as e:
            print(f"Collaborative recommendation failed: {e}")
        
        return content_recs, collaborative_recs

    def _enhanced_scoring(self, user_id: int, candidates: pd.DataFrame) -> Dict[int, float]:
        """Enhanced scoring with multiple signal integration"""
        scores = {}
        content_weight, collaborative_weight = self._adaptive_weights_enhanced(user_id)
        
        for book_id in candidates['book_id'].values:
            total_score = 0.0
            weight_sum = 0.0
            
            # Content-based score
            try:
                content_profile = self.content_recommender._get_user_profile_enhanced(user_id)
                if content_profile is not None and book_id in self.content_recommender.book_id_to_idx:
                    book_idx = self.content_recommender.book_id_to_idx[book_id]
                    book_features = self.content_recommender.feature_matrix[book_idx]
                    content_score = float(np.dot(content_profile, book_features) / 
                                        (np.linalg.norm(content_profile) * np.linalg.norm(book_features) + 1e-10))
                    content_score = max(0.0, content_score)
                    
                    total_score += content_weight * content_score
                    weight_sum += content_weight
            except Exception:
                pass
            
            # Collaborative score
            try:
                collab_rating = self.collaborative_recommender.predict_rating(user_id, book_id)
                if collab_rating and collab_rating > 0:
                    collab_score = (collab_rating - 1.0) / 4.0  # Normalize to 0-1
                    total_score += collaborative_weight * collab_score
                    weight_sum += collaborative_weight
            except Exception:
                pass
            
            # Popularity and quality boost
            try:
                book_info = candidates[candidates['book_id'] == book_id].iloc[0]
                popularity_score = min(1.0, np.log1p(book_info.get('ratings_count', 1)) / 10.0)
                quality_score = book_info.get('average_rating', 3.5) / 5.0
                
                # Small boost from popularity and quality
                boost = 0.1 * (0.6 * quality_score + 0.4 * popularity_score)
                total_score += boost
                weight_sum += 0.1
            except Exception:
                pass
            
            # Final score
            if weight_sum > 0:
                scores[book_id] = total_score / weight_sum
            else:
                scores[book_id] = 0.0
        
        return scores

    def _diversify_enhanced(self, user_id: int, scores: Dict[int, float]) -> Dict[int, float]:
        """Enhanced diversification with multiple diversity measures"""
        try:
            # Get user's reading history
            user_books = self.user_to_items.get(user_id, {})
            
            if not user_books:
                return scores
            
            # Get rated books information
            rated_book_ids = list(user_books.keys())
            rated_books = self.books_df[self.books_df['book_id'].isin(rated_book_ids)]
            
            # Extract user's preferences
            liked_authors = set()
            liked_years = set()
            avg_liked_rating = 0
            
            for _, book in rated_books.iterrows():
                book_id = book['book_id']
                user_rating = user_books.get(book_id, 3)
                
                if user_rating >= 4:  # Liked books
                    # Extract authors
                    authors = str(book.get('authors', ''))
                    if authors and authors != 'nan':
                        liked_authors.update([a.strip() for a in authors.split(',')[:2]])  # Top 2 authors
                    
                    # Extract years
                    year = book.get('original_publication_year')
                    if year and not pd.isna(year):
                        liked_years.add(int(year))
            
            # Apply diversity adjustments
            diversified_scores = scores.copy()
            
            for book_id in list(scores.keys()):
                if book_id not in self.book_rows.index:
                    continue
                
                book_info = self.book_rows.loc[book_id]
                diversity_factor = 1.0
                
                # Author diversity
                book_authors = set([a.strip() for a in str(book_info.get('authors', '')).split(',')[:2]])
                if liked_authors and book_authors & liked_authors:
                    diversity_factor *= 0.92  # Slight penalty for same authors
                
                # Temporal diversity
                book_year = book_info.get('original_publication_year')
                if book_year and not pd.isna(book_year) and liked_years:
                    year_distances = [abs(int(book_year) - liked_year) for liked_year in liked_years]
                    min_distance = min(year_distances)
                    if min_distance > 15:  # Different era
                        diversity_factor *= 1.08  # Small boost for temporal diversity
                
                diversified_scores[book_id] *= diversity_factor
            
            return diversified_scores
            
        except Exception as e:
            return scores

    def recommend(self, user_id: int, n_recommendations: int = 10) -> pd.DataFrame:
        """Enhanced hybrid recommendation with optimized performance"""
        try:
            # Check cache first
            cache_key = f"{user_id}_{n_recommendations}"
            if cache_key in self._recommendation_cache:
                return self._recommendation_cache[cache_key]
            
            # Get recommendations from both algorithms
            content_recs, collaborative_recs = self._get_recommendations_from_algorithms(
                user_id, n_recommendations
            )
            
            # Create candidate pool
            all_candidates = []
            if not content_recs.empty:
                all_candidates.append(content_recs)
            if not collaborative_recs.empty:
                all_candidates.append(collaborative_recs)
            
            if not all_candidates:
                # Fallback to popular books
                fallback = self.books_df.nlargest(n_recommendations, 'ratings_count').copy()
                fallback['hybrid_score'] = 0.1
                return fallback
            
            # Combine candidates
            candidates = pd.concat(all_candidates, ignore_index=True)
            candidates = candidates.drop_duplicates(subset=['book_id'])
            
            # Merge with complete book information
            candidates = candidates.merge(self.books_df, on='book_id', how='left', suffixes=('', '_full'))
            
            # Remove books already rated by user
            seen_books = set(self.user_to_items.get(user_id, {}).keys())
            candidates = candidates[~candidates['book_id'].isin(seen_books)]
            
            if candidates.empty:
                # Fallback
                fallback = self.books_df.nlargest(n_recommendations, 'ratings_count').copy()
                fallback['hybrid_score'] = 0.1
                return fallback
            
            # Enhanced scoring
            scores = self._enhanced_scoring(user_id, candidates)
            
            # Apply diversification
            scores = self._diversify_enhanced(user_id, scores)
            
            # Final ranking
            ranked_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            top_book_ids = [book_id for book_id, _ in ranked_items[:n_recommendations]]
            
            # Create final result
            result = self.books_df[self.books_df['book_id'].isin(top_book_ids)].copy()
            result['hybrid_score'] = result['book_id'].map(scores)
            result = result.sort_values('hybrid_score', ascending=False)
            
            # Cache result
            self._recommendation_cache[cache_key] = result
            
            return result.head(n_recommendations)
            
        except Exception as e:
            print(f"Hybrid recommendation error: {e}")
            # Return fallback recommendations
            fallback = self.books_df.nlargest(n_recommendations, 'ratings_count').copy()
            fallback['hybrid_score'] = 0.1
            return fallback
        
    def recommend_from_query(self, query_text: str, n_recommendations: int = 10):
        print("query hybrid mode")
        print(query_text)
        """
        Query-based hybrid recommendations:
        - use content_recommender.recommend_from_query
        - use collaborative_recommender.recommend_from_query
        - combine candidates and score them (no logged-in user profile assumed)
        """
        try:
            # Get content and collaborative candidates from query mode
            content_recs = pd.DataFrame()
            collab_recs = pd.DataFrame()

            try:
                content_recs = self.content_recommender.recommend_from_query(query_text, n_recommendations * 3)
            except Exception:
                print("bb")
                content_recs = pd.DataFrame()

            try:
                collab_recs = self.collaborative_recommender.recommend_from_query(query_text, n_recommendations * 3)
            except Exception:
                print("bb1")
                collab_recs = pd.DataFrame()

            # If both empty, fallback to popular
            if (content_recs.empty) and (collab_recs.empty):
                fallback = self.books_df.nlargest(n_recommendations, 'ratings_count').copy()
                fallback['hybrid_score'] = 0.1
                print("bb2")
                return fallback.head(n_recommendations)

            # Combine candidate pool
            candidates = pd.concat([content_recs, collab_recs], ignore_index=True).drop_duplicates(subset=['book_id'])
            candidates = candidates.merge(self.books_df, on='book_id', how='left', suffixes=('', '_full'))

            # Scoring: because we don't have logged-in user adaptive weights, use content-heavy default
            content_weight = 0.6
            collaborative_weight = 0.4

            scores = {}
            for _, row in candidates.iterrows():
                book_id = int(row['book_id'])
                total_score = 0.0
                weight_sum = 0.0

                # content signal (if present)
                cscore = 0.0
                if 'content_similarity' in row.index and not pd.isnull(row.get('content_similarity', None)):
                    cscore = float(row.get('content_similarity', 0.0))
                total_score += content_weight * cscore
                weight_sum += content_weight

                # collaborative signal (if present)
                cfscore = 0.0
                if 'score' in row.index and not pd.isnull(row.get('score', None)):
                    cfscore = float(row.get('score', 0.0))
                elif 'ensemble_score' in row.index and not pd.isnull(row.get('ensemble_score', None)):
                    cfscore = float(row.get('ensemble_score', 0.0))
                total_score += collaborative_weight * cfscore
                weight_sum += collaborative_weight

                # popularity & quality small boost
                try:
                    popularity_score = min(1.0, np.log1p(row.get('ratings_count', 1)) / 10.0)
                    quality_score = row.get('average_rating', 3.5) / 5.0
                    boost = 0.08 * (0.6 * quality_score + 0.4 * popularity_score)
                    total_score += boost
                    weight_sum += 0.08
                except Exception:
                    print("bb3")
                    pass

                # final normalized
                scores[book_id] = (total_score / weight_sum) if weight_sum > 0 else 0.0

            # Diversify similar to existing _diversify_enhanced (use simplified penalty)
            for book_id in list(scores.keys()):
                try:
                    book_info = self.book_rows.loc[book_id]
                    # slight penalty if same author appears in top seeds
                    authors = set([a.strip() for a in str(book_info.get('authors', '')).split(',')[:2]])
                    # simple author frequency check
                    authors_count = sum(1 for bid in candidates['book_id'] for auth in [a.strip() for a in str(candidates.loc[candidates['book_id'] == bid, 'authors'].values[0]).split(',')[:2]] if auth in authors)
                    if authors_count > 3:
                        scores[book_id] *= 0.94
                except Exception:
                    print("bb4")
                    continue

            # Rank and return
            ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n_recommendations]
            book_ids = [b for b, _ in ranked]
            result = self.books_df[self.books_df['book_id'].isin(book_ids)].copy()
            result['hybrid_score'] = result['book_id'].map(scores)
            result = result.sort_values('hybrid_score', ascending=False)
            print("bb5")
            return result.head(n_recommendations)

        except Exception as e:
            print(f"[Hybrid] recommend_from_query error: {e}")
            fallback = self.books_df.nlargest(n_recommendations, 'ratings_count').copy()
            fallback['hybrid_score'] = 0.1
            print("bb6")
            return fallback.head(n_recommendations)


    def predict_rating(self, user_id: int, book_id: int) -> float:
        """Enhanced rating prediction with ensemble methods"""
        try:
            content_weight, collaborative_weight = self._adaptive_weights_enhanced(user_id)
            
            predictions = []
            weights = []
            
            # Content-based prediction
            try:
                content_profile = self.content_recommender._get_user_profile_enhanced(user_id)
                if content_profile is not None and book_id in self.content_recommender.book_id_to_idx:
                    book_idx = self.content_recommender.book_id_to_idx[book_id]
                    book_features = self.content_recommender.feature_matrix[book_idx]
                    similarity = np.dot(content_profile, book_features) / (
                        np.linalg.norm(content_profile) * np.linalg.norm(book_features) + 1e-10
                    )
                    content_pred = 1.0 + 4.0 * max(0, min(1, similarity))
                    predictions.append(content_pred)
                    weights.append(content_weight)
            except Exception:
                pass
            
            # Collaborative prediction
            try:
                collab_pred = self.collaborative_recommender.predict_rating(user_id, book_id)
                if collab_pred and collab_pred > 0:
                    predictions.append(float(collab_pred))
                    weights.append(collaborative_weight)
            except Exception:
                pass
            
            # Ensemble prediction
            if predictions:
                weighted_pred = sum(p * w for p, w in zip(predictions, weights)) / sum(weights)
                return max(1.0, min(5.0, weighted_pred))
            else:
                # Fallback to global average with book bias
                try:
                    book_info = self.book_rows.loc[book_id] if book_id in self.book_rows.index else None
                    if book_info is not None:
                        return max(1.0, min(5.0, float(book_info.get('average_rating', 3.5))))
                    else:
                        return 3.5
                except:
                    return 3.5
                    
        except Exception as e:
            return 3.5

    def explain_recommendation(self, user_id: int, book_id: int) -> dict:
        """Enhanced recommendation explanation with detailed reasoning"""
        try:
            content_weight, collaborative_weight = self._adaptive_weights_enhanced(user_id)
            
            book_info = self.book_rows.loc[book_id] if book_id in self.book_rows.index else None
            if book_info is None:
                return {"error": "Book not found"}
            
            explanation = {
                'weights': {
                    'content_based': round(content_weight, 3),
                    'collaborative': round(collaborative_weight, 3)
                },
                'scores': {},
                'reasons': {
                    'content_based': [],
                    'collaborative': [],
                    'popularity': None,
                    'quality': None
                }
            }
            
            # Content-based explanation
            try:
                content_profile = self.content_recommender._get_user_profile_enhanced(user_id)
                if content_profile is not None and book_id in self.content_recommender.book_id_to_idx:
                    book_idx = self.content_recommender.book_id_to_idx[book_id]
                    similarity = self.content_recommender.similarity_matrix[book_idx]
                    
                    # Find most similar books user has rated
                    user_books = self.user_to_items.get(user_id, {})
                    similar_rated = []
                    
                    for rated_book_id, rating in user_books.items():
                        if rating >= 4 and rated_book_id in self.content_recommender.book_id_to_idx:
                            rated_idx = self.content_recommender.book_id_to_idx[rated_book_id]
                            sim_score = similarity[rated_idx]
                            if sim_score > 0.3:  # Meaningful similarity
                                rated_book_info = self.books_df[self.books_df['book_id'] == rated_book_id]
                                if not rated_book_info.empty:
                                    similar_rated.append((
                                        rated_book_info.iloc[0]['title'],
                                        sim_score
                                    ))
                    
                    # Sort by similarity and take top 2
                    similar_rated.sort(key=lambda x: x[1], reverse=True)
                    for title, sim in similar_rated[:2]:
                        explanation['reasons']['content_based'].append(
                            f"Similar to '{title}' (similarity: {sim:.3f})"
                        )
                        
            except Exception as e:
                explanation['reasons']['content_based'].append("Content analysis unavailable")
            
            # Collaborative explanation
            try:
                if user_id in self.collaborative_recommender.user_to_idx:
                    explanation['reasons']['collaborative'].append(
                        f"Recommended by users with similar reading patterns"
                    )
                    
                    # Add predicted rating
                    pred_rating = self.collaborative_recommender.predict_rating(user_id, book_id)
                    if pred_rating:
                        explanation['scores']['predicted_rating'] = round(pred_rating, 2)
                        
            except Exception:
                explanation['reasons']['collaborative'].append("Collaborative analysis unavailable")
            
            # Popularity and quality
            try:
                ratings_count = int(book_info.get('ratings_count', 0))
                avg_rating = float(book_info.get('average_rating', 0))
                
                explanation['reasons']['popularity'] = f"Popular choice ({ratings_count:,} ratings)"
                explanation['reasons']['quality'] = f"Highly rated (avg: {avg_rating:.2f}/5.0)"
                
            except Exception:
                pass
            
            return explanation
            
        except Exception as e:
            return {"error": f"Explanation failed: {str(e)}"}

    def get_performance_stats(self):
        """Get hybrid recommender performance statistics"""
        return {
            'cache_size': len(self._recommendation_cache),
            'weight_cache_size': len(self._user_weight_cache),
            'total_users': len(self.user_to_items),
            'total_books': len(self.books_df),
            'avg_user_ratings': np.mean([len(ratings) for ratings in self.user_to_items.values()]),
            'content_recommender_ready': self.content_recommender is not None,
            'collaborative_recommender_ready': self.collaborative_recommender is not None
        }