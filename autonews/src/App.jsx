import { useState, useEffect } from 'react';
import './App.css';

function App() {
  const [searchTerm, setSearchTerm] = useState('');
  const [displayedSearchTerm, setDisplayedSearchTerm] = useState('');
  const [isMoved, setIsMoved] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [results, setResults] = useState({ latest_news: [], stock_updates: [], average_sentiment: {} });
  const [error, setError] = useState('');

  const fetchResults = async (company) => {
    setIsLoading(true);
    setError('');
    setResults({ latest_news: [], stock_updates: [], average_sentiment: { average: "" } }); // Reset sentiment
  
    try {
      const response = await fetch(`http://127.0.0.1:5000/search?company=` + encodeURIComponent(company));
      if (!response.ok) {
        throw new Error('Failed to fetch results');
      }
  
      const data = await response.json();
      setResults(data);
      setDisplayedSearchTerm(company); // Update only when fetch is successful
    } catch (err) {
      setError('Error fetching results. Please try again.');
      console.error('Error:', err);
    } finally {
      setIsLoading(false);
    }
  };
  

  const handleKeyDown = (event) => {
    if (event.key === 'Enter' && searchTerm) {
      setIsMoved(true);
      fetchResults(searchTerm);
    }
  };

  return (
    <div style={{ backgroundColor: 'black', height: '100vh', display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', color: 'white', overflow: 'hidden' }}>
      <div style={{ position: 'absolute', textAlign: 'center', top: '40%', left: isMoved ? '12.5%' : '50%', transform: isMoved ? 'translateX(0)' : 'translateX(-50%)', fontSize: '50px', fontWeight: 'bold', transition: 'left 0.5s ease' }}>Auto News</div>
      
      <input
        type="text"
        placeholder="Search a company"
        value={searchTerm}
        onChange={(e) => setSearchTerm(e.target.value)}
        onKeyDown={handleKeyDown}
        style={{ padding: '7px', fontSize: '20px', borderRadius: '20px', border: 'none', outline: 'none', width: '300px', height: '50px', textAlign: 'center', position: 'absolute', left: isMoved ? '10%' : '50%', top: '50%', transform: isMoved ? 'translateX(0)' : 'translateX(-50%)', transition: 'left 0.5s ease, top 0.5s ease', backgroundColor: 'white', color: 'black' }}
      />
      {isMoved && (
        <div
          style={{
            position: "absolute",
            left: "10%",
            top: "60%",
            fontSize: "50px",
            textAlign: "center",
            display: "flex",
            justifyContent: "center",
            alignItems: "center",
            width: "300px",
            height: "50px",
            lineHeight: "50px", 
          }}
        >
          {results.average_sentiment.average ? (
            results.average_sentiment.average === "Positive"
              ? "Buy!"
              : results.average_sentiment.average === "Negative"
              ? "Sell!"
              : "Hold!"
          ) : (
            " "
          )}
        </div>
        
      )}


      {isMoved && (
        <div style={{ position: 'absolute', right: '0', top: '0', width: '50%', height: '100vh', overflowY: 'auto', padding: '20px', boxSizing: 'border-box', backgroundColor: 'black' }}>
          {isLoading ? (
            <div style={{ textAlign: 'center', marginTop: '100px' }}>Loading...</div>
          ) : error ? (
            <div style={{ color: 'red', textAlign: 'center', marginTop: '100px' }}>{error}</div>
          ) : (
            <div>
              <h2 style={{ borderBottom: '1px solid #333', paddingBottom: '10px', marginTop: '80px', display: 'flex', justifyContent: 'space-between' }}>
                Latest News for "{displayedSearchTerm}" {/* Only updates after a successful query */}
                <span style={{ color: results.average_sentiment.latest_news === "Positive" ? "#2cb9ff" : results.average_sentiment.latest_news === "Neutral" ? "gray" : "red" }}>
                  {results.average_sentiment.latest_news}
                </span>
              </h2>
              {results.latest_news.length > 0 ? (
                results.latest_news.map((item, index) => (
                  <div key={`news-${index}`} style={{ marginBottom: '20px', padding: '15px', backgroundColor: 'black', borderRadius: '8px', textAlign: 'left' }}>
                    <h3 style={{ margin: '0 0 10px 0' }}>{item.title?.trim() ? item.title : "No title found"}</h3>
                    <h3 style={{ margin: '0 0 10px 0', color: item.sentiment === "Positive" ? "#2cb9ff" : item.sentiment === "Neutral" ? "gray" : "red" }}>{item.sentiment}</h3>
                    <p style={{ margin: '0', color: '#aaa' }}>{item.description?.trim() ? item.description : "No description found"}</p>
                    <a href={item.url} target="_blank" rel="noopener noreferrer" style={{ color: '#92C4FF', textDecoration: 'none', display: 'block', marginTop: '10px' }}>Read more</a>
                  </div>
                ))
              ) : (
                <p>No news results found</p>
              )}
              
              <h2 style={{ borderBottom: '1px solid #333', paddingBottom: '10px', marginTop: '30px', display: 'flex', justifyContent: 'space-between' }}>
                Stock Updates
                <span style={{ color: results.average_sentiment.stock_updates === "Positive" ? "#2cb9ff" : results.average_sentiment.stock_updates === "Neutral" ? "gray" : "red" }}>
                  {results.average_sentiment.stock_updates}
                </span>
              </h2>
              {results.stock_updates.length > 0 ? (
                results.stock_updates.map((item, index) => (
                  <div key={`stock-${index}`} style={{ marginBottom: '20px', padding: '15px', backgroundColor: 'black', borderRadius: '8px', textAlign: 'left' }}>
                    <h3 style={{ margin: '0 0 10px 0' }}>{item.title?.trim() ? item.title : "No title found"}</h3>
                    <h3 style={{ margin: '0 0 10px 0', color: item.sentiment === "Positive" ? "#2cb9ff" : item.sentiment === "Neutral" ? "gray" : "red" }}>{item.sentiment}</h3>
                    <p style={{ margin: '0', color: '#aaa' }}>{item.description?.trim() ? item.description : "No description found"}</p>
                    <a href={item.url} target="_blank" rel="noopener noreferrer" style={{ color: '#4a9eff', textDecoration: 'none', display: 'block', marginTop: '10px' }}>Read more</a>
                  </div>
                ))
              ) : (
                <p>No stock updates found</p>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default App;
