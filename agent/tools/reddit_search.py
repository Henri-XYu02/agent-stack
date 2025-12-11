from typing import List, Dict, Optional
import requests
import time


def search_reddit(query: str, subreddit: Optional[str] = None, max_results: int = 5, sort: str = "relevance") -> str:
    """Search Reddit posts using the Reddit JSON API (no authentication required).

    Args:
        query: The search query
        subreddit: Optional subreddit to search within (e.g., "python", "programming")
        max_results: Maximum number of results to return (default: 5)
        sort: Sort method - "relevance", "hot", "top", "new", "comments" (default: "relevance")

    Returns:
        Formatted string with Reddit search results including titles, scores, and content
    """
    try:
        # More comprehensive headers to avoid blocking
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1"
        }

        # Build the search URL - use old.reddit.com as fallback
        if subreddit:
            url = f"https://old.reddit.com/r/{subreddit}/search.json"
        else:
            url = "https://old.reddit.com/search.json"

        params = {
            "q": query,
            "limit": max_results,
            "sort": sort,
            "restrict_sr": "true" if subreddit else "false",
            "raw_json": 1,
            "t": "all"  # time filter
        }

        # Add a small delay to be respectful
        time.sleep(0.5)

        # Make the request with extended timeout
        response = requests.get(url, headers=headers, params=params, timeout=15)
        
        # Check for rate limiting or blocking
        if response.status_code == 429:
            return "Rate limited by Reddit. Please try again in a few moments."
        elif response.status_code == 403:
            return "Access forbidden by Reddit. This may be due to IP restrictions or rate limiting."
        
        response.raise_for_status()

        data = response.json()

        # Extract posts
        posts = data.get("data", {}).get("children", [])

        if not posts:
            return f"No Reddit results found for query: '{query}'"

        # Format results
        formatted_results = []
        for i, post in enumerate(posts[:max_results], 1):
            post_data = post.get("data", {})

            title = post_data.get("title", "No title")
            author = post_data.get("author", "Unknown")
            subreddit_name = post_data.get("subreddit", "")
            score = post_data.get("score", 0)
            num_comments = post_data.get("num_comments", 0)
            url = f"https://www.reddit.com{post_data.get('permalink', '')}"
            selftext = post_data.get("selftext", "")
            created_utc = post_data.get("created_utc", 0)

            result = f"{i}. [{subreddit_name}] {title}\n"
            result += f"   Author: u/{author} | Score: {score} | Comments: {num_comments}\n"
            result += f"   URL: {url}\n"

            if selftext:
                # Truncate long posts
                if len(selftext) > 300:
                    selftext = selftext[:300] + "..."
                result += f"   Content: {selftext}\n"

            formatted_results.append(result)

        return "\n".join(formatted_results)

    except requests.exceptions.HTTPError as e:
        return f"HTTP Error: {e.response.status_code} - {e.response.reason}"
    except requests.exceptions.RequestException as e:
        return f"Error searching Reddit: {str(e)}"
    except Exception as e:
        return f"Error processing Reddit results: {str(e)}"


def get_subreddit_posts(subreddit: str, category: str = "hot", max_results: int = 5) -> str:
    """Get posts from a subreddit by category (hot, new, top, rising).
    
    This is an alternative that doesn't use search and may work better.
    
    Args:
        subreddit: The subreddit name (e.g., "python", "programming")
        category: Category - "hot", "new", "top", "rising" (default: "hot")
        max_results: Maximum number of results to return (default: 5)
    
    Returns:
        Formatted string with Reddit posts
    """
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }

        url = f"https://old.reddit.com/r/{subreddit}/{category}.json"
        params = {"limit": max_results, "raw_json": 1}

        time.sleep(0.5)
        response = requests.get(url, headers=headers, params=params, timeout=15)
        
        if response.status_code == 429:
            return "Rate limited by Reddit. Please try again in a few moments."
        elif response.status_code == 403:
            return "Access forbidden by Reddit."
            
        response.raise_for_status()
        data = response.json()

        posts = data.get("data", {}).get("children", [])
        
        if not posts:
            return f"No posts found in r/{subreddit}"

        formatted_results = []
        for i, post in enumerate(posts[:max_results], 1):
            post_data = post.get("data", {})
            
            title = post_data.get("title", "No title")
            author = post_data.get("author", "Unknown")
            score = post_data.get("score", 0)
            num_comments = post_data.get("num_comments", 0)
            url = f"https://www.reddit.com{post_data.get('permalink', '')}"
            selftext = post_data.get("selftext", "")
            
            result = f"{i}. {title}\n"
            result += f"   Author: u/{author} | Score: {score} | Comments: {num_comments}\n"
            result += f"   URL: {url}\n"
            
            if selftext:
                if len(selftext) > 300:
                    selftext = selftext[:300] + "..."
                result += f"   Content: {selftext}\n"
            
            formatted_results.append(result)
        
        return "\n".join(formatted_results)
        
    except Exception as e:
        return f"Error fetching subreddit posts: {str(e)}"


if __name__ == "__main__":
    # Test the function
    print("Test 1: General search")
    results = search_reddit("python programming", max_results=3)
    print(results)
    print("\n" + "="*80 + "\n")

    print("Test 2: Subreddit-specific search")
    results = search_reddit("best practices", subreddit="python", max_results=3)
    print(results)
    print("\n" + "="*80 + "\n")
    
    print("Test 3: Get hot posts (alternative method)")
    results = get_subreddit_posts("python", category="hot", max_results=3)
    print(results)