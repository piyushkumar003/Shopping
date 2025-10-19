// src/Analytics.js
import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

// API base URL for our FastAPI backend
const API_URL = 'http://127.0.0.1:8000';

const Analytics = () => {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Fetch data from our /analytics endpoint
    const fetchData = async () => {
      try {
        const response = await axios.get(`${API_URL}/analytics`);
        setData(response.data);
        setLoading(false);
      } catch (error) {
        console.error("Error fetching analytics data:", error);
        setLoading(false);
      }
    };
    fetchData();
  }, []); // Empty array means this runs once on component load

  if (loading) {
    return <div className="analytics-container"><h2>Loading Analytics...</h2></div>;
  }
  
  if (!data || data.error) {
    return <div className="analytics-container"><h2>Could not load analytics data.</h2></div>;
  }

  return (
    <div className="analytics-container">
      <h2>Product Analytics</h2>
      
      {/* Chart 1: Top 10 Categories */}
      <h3>Top 10 Product Categories</h3>
      <ChartComponent data={data.category_counts} dataKey="value" />
      
      {/* Chart 2: Top 10 Brands */}
      <h3>Top 10 Product Brands</h3>
      <ChartComponent data={data.brand_counts} dataKey="value" />
      
      {/* Chart 3: Top 10 Materials */}
      <h3>Top 10 Product Materials</h3>
      <ChartComponent data={data.material_counts} dataKey="value" />
    </div>
  );
};

// Reusable Bar Chart Component
const ChartComponent = ({ data, dataKey }) => (
  <div style={{ width: '100%', height: 400, marginBottom: '20px' }}>
    <ResponsiveContainer>
      <BarChart
        data={data}
        layout="vertical" // Makes it a horizontal bar chart, better for long labels
        margin={{ top: 5, right: 30, left: 100, bottom: 5 }}
      >
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis type="number" />
        <YAxis dataKey="name" type="category" width={150} />
        <Tooltip />
        <Legend />
        <Bar dataKey={dataKey} fill="#8884d8" name="Product Count" />
      </BarChart>
    </ResponsiveContainer>
  </div>
);

export default Analytics;