import os, time

try:
    from serpapi import GoogleSearch  # type: ignore
    _HAS_GOOGLESEARCH = True
except Exception:
    import serpapi  # type: ignore
    GoogleSearch = None
    _HAS_GOOGLESEARCH = False


class GoogleSearchAPI:
    def __init__(self, api_key=None):
        # 
        self.api_key = api_key or os.getenv("SERPAPI_API_KEY")

    def _call_search(self, params: dict):
        if _HAS_GOOGLESEARCH and GoogleSearch is not None:
            search = GoogleSearch(params)
            # GoogleSearch from some versions exposes get_dict()
            return getattr(search, "get_dict", lambda: {})()
        else:
            # Fallback to the installed serpapi client's search function
            try:
                res = serpapi.search(params)
            except Exception:
                # also support calling with kwargs
                res = serpapi.search(**params)

            # SerpResults exposes as_dict(), otherwise behave like a dict
            if hasattr(res, "as_dict"):
                return res.as_dict()
            if isinstance(res, dict):
                return res
            return {}

    def retrieve(self, query, k=5):
        start = time.perf_counter()  

        params = {
            "engine": "google",
            "q": query,
            "api_key": self.api_key,
            "num": k,
        }

        results = self._call_search(params)

        snippets = []

        # Extract snippets from organic results
        if isinstance(results, dict) and "organic_results" in results:
            for res in results["organic_results"][:k]:
                snippet = res.get("snippet", "")
                if snippet:
                    snippets.append(snippet)

        elapsed = time.perf_counter() - start 

        return snippets, elapsed 

    def retrieve_chunk_id(self, query, k=5):
        start = time.perf_counter()

        params = {
            "engine": "google",
            "q": query,
            "api_key": self.api_key,
            "num": k,
        }

        results = self._call_search(params)

        ids = []
        if isinstance(results, dict) and "organic_results" in results:
            num = min(k, len(results["organic_results"]))
            ids = [i for i in range(num)]

        elapsed = time.perf_counter() - start 

        return ids, elapsed