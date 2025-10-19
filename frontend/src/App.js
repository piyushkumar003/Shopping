import React, { useState } from 'react';
import { BrowserRouter as Router, Routes, Route, NavLink } from 'react-router-dom';
import axios from 'axios';
import Analytics from './Analytics';
import './App.css';

function App() {
  return (
    <Router>
      <div className="App">
        <nav className="navbar">
          <h1>AI Furniture Recommender</h1>
          <div className="nav-links">
            <NavLink to="/">Recommendations</NavLink>
            <NavLink to="/analytics">Analytics</NavLink>
          </div>
        </nav>
        <main>
          <Routes>
            <Route path="/" element={<RecommendationPage />} />
            <Route path="/analytics" element={<Analytics />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}

const RecommendationPage = () => {
  const [query, setQuery] = useState('');
  const [chat, setChat] = useState([{ from: 'bot', text: 'Hi! What kind of furniture are you looking for today?' }]);
  const [loading, setLoading] = useState(false);

  const API_URL = 'https://piyush.mankiratsingh.com';

  const handleSend = async () => {
    if (!query.trim()) return;

    const userMessage = { from: 'user', text: query };
    setChat(prevChat => [...prevChat, userMessage]);
    setQuery('');
    setLoading(true);
    
    try {
      const response = await axios.post(`${API_URL}/recommend`, { query: query });
      
      const botMessage = { 
        from: 'bot', 
        products: response.data.recommendations 
      };
      setChat(prevChat => [...prevChat, userMessage, botMessage]);
      
    } catch (error) {
      console.error("Error fetching recommendations:", error);
      const errorMessage = { from: 'bot', text: "Sorry, I couldn't fetch recommendations right now. Please check if the backend is running." };
      setChat(prevChat => [...prevChat, userMessage, errorMessage]);
    }
    setLoading(false);
  };

  return (
    <div className="chat-container">
      <div className="chat-box">
        {chat.map((msg, index) => (
          <div key={index} className={`chat-message ${msg.from}`}>
            {msg.text && <p>{msg.text}</p>}
            {msg.products && (
              <div className="product-list">
                <h3>Here are some recommendations I found for you:</h3>
                {msg.products.map((product, idx) => (
                  <ProductCard key={product.title + idx} product={product} />
                ))}
              </div>
            )}
          </div>
        ))}
        {loading && <div className="chat-message bot"><p>Finding the perfect items for you...</p></div>}
      </div>
      <div className="input-area">
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyPress={(e) => e.key === 'Enter' && !loading && handleSend()}
          placeholder="e.g., 'a comfy brown chair for reading'"
          disabled={loading}
        />
        <button onClick={handleSend} disabled={loading}>Send</button>
      </div>
    </div>
  );
};

const ProductCard = ({ product }) => {
  // Fallback image in case the original fails to load
  const placeholderImg = `https://placehold.co/120x120/e9ecef/495057?text=No+Image`;
  return (
    <div className="product-card">
      <img 
        src={product.image_url || placeholderImg} 
        alt={product.title} 
        className="product-image" 
        onError={(e) => { e.target.onerror = null; e.target.src=placeholderImg; }}
      />
      <div className="product-info">
        <h4>{product.title}</h4>
        <p className="product-price">${product.price ? product.price.toFixed(2) : 'N/A'}</p>
        <p className="product-description">{product.generated_description}</p>
      </div>
    </div>
  );
};

export default App;
