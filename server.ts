import { GoogleGenAI, ThinkingLevel } from "@google/genai";
import express from "express";
import path from "path";
import fs from 'fs';
import { createServer as createViteServer } from "vite";
import { initializeApp, applicationDefault } from 'firebase-admin/app';
import { getFirestore } from 'firebase-admin/firestore';
import multer from 'multer';

// Storage setup for multer
const upload = multer({ dest: 'knowledge_base/' });

// Initialize Firebase Admin
const firebaseApp = initializeApp({
  credential: applicationDefault()
});
const db = getFirestore(firebaseApp);

const app = express();
const PORT = 3000;

// Initialize Gemin
const ai = new GoogleGenAI({
  apiKey: process.env.GEMINI_API_KEY,
  httpOptions: { headers: { 'User-Agent': 'aistudio-build' } }
});

// Helper to fetch weather
async function getWeatherData(city: string) {
  try {
    // 1. Geocode
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 5000); // 5s timeout
    
    const geoResponse = await fetch(`https://geocoding-api.open-meteo.com/v1/search?name=${encodeURIComponent(city)}`, { signal: controller.signal });
    clearTimeout(timeoutId);
    
    const geoData = await geoResponse.json();
    if (!geoData.results) return null;
    const { latitude, longitude, name, admin1, country } = geoData.results[0];

    // 2. Weather
    const controllerWeather = new AbortController();
    const timeoutIdWeather = setTimeout(() => controllerWeather.abort(), 5000); // 5s timeout
    
    const weatherResponse = await fetch(`https://api.open-meteo.com/v1/forecast?latitude=${latitude}&longitude=${longitude}&current=temperature_2m,relative_humidity_2m,wind_speed_10m`, { signal: controllerWeather.signal });
    clearTimeout(timeoutIdWeather);
    const weatherData = await weatherResponse.json();
    
    return {
        location: `${name}, ${admin1}, ${country}`,
        temp: weatherData.current.temperature_2m,
        humidity: weatherData.current.relative_humidity_2m,
        wind: weatherData.current.wind_speed_10m
    };
  } catch (e) {
    console.error(e);
    return null;
  }
}

app.use(express.json());

// Proactive Memory Helpers
async function extractAndSaveMemories(userId: string, message: string, reply: string) {
    try {
        const extractionPrompt = `Analyze the following interaction and extract any personal preferences, interests, or important facts about the user that should be remembered. 
        Message: "${message}" 
        Reply: "${reply}"
        If something important exists, return just the memory string. If nothing important, return "NONE".`;
        
        const geminiResponse = await ai.models.generateContent({
            model: "gemini-3.5-flash",
            contents: extractionPrompt
        });
        
        const memory = geminiResponse.text ? geminiResponse.text.trim() : "NONE";
        if (memory !== "NONE") {
            await db.collection('users').doc(userId).collection('memories').add({
                userId,
                content: memory,
                createdAt: new Date().toISOString()
            });
        }
    } catch (e) {
        console.error("Memory extraction failed:", e);
    }
}

// Helper for RAG + Search Grounding
async function answerWithFallbackChain(message: string, userId: string): Promise<string> {
  const query = message;

  // 1. Try RAG Context
  const snapshot = await db.collection('knowledge_base')
      .limit(5)
      .get();
  
  let RAGContext = "";
  if (!snapshot.empty) {
      const docs = snapshot.docs.map(d => d.data().content);
      RAGContext = `RAG Knowledge Base Context:\n${docs.join('\n\n')}`;
  }

  // 2. Perform explicit Search Task
  console.log(`Performing search for: ${query}`);
  const searchResult = await ai.models.generateContent({ 
    model: "gemini-3.8-flash", 
    contents: `Search for information regarding: "${query}"`,
    tools: [{ googleSearch: {} }],
    config: {
        toolConfig: { includeServerSideToolInvocations: true }
    }
  });

  // 3. Synthesis Prompt using search results and RAG context
  const finalPrompt = `
  You are an advanced AI assistant, Nexora AI.
  
  **Task:** Answer the user's question accurately, synthesized from the provided RAG context, the search results, and your general knowledge.
  
  **Constraints:**
  - Be clear, accurate, and concise.
  - If uncertain, state clearly.
  - Never invent facts.
  - **Cite sources** for any information retrieved via search.
  
  **Strict Output Format:**
  Answer:
  [Final synthesized answer with embedded citations e.g., [1], [2]]
  
  Sources:
  [List of sources with URLs]
  
  **Context:**
  ${RAGContext}
  
  **Search Findings:**
  ${searchResult.text}
  
  **Question:** "${query}"
  `;

  // Final Synthesis
  const finalResult = await ai.models.generateContent({ 
    model: "gemini-3.8-flash", 
    contents: finalPrompt
  });

  return finalResult.text || "I'm sorry, I couldn't synthesize an answer to that.";
}

// Update /api/chat to use this chain
app.post("/api/chat", async (req, res) => {
  const { message, userId, mode, language } = req.body;
  
  try {
    const reply = await answerWithFallbackChain(message, userId);
    
    // Extract memories (keeping existing functionality)
    await extractAndSaveMemories(userId, message, reply);
    
    res.json({ reply: reply });
  } catch (error: any) {
    console.error("Error in /api/chat:", error);
    const errorMessage = error instanceof Error ? error.message : String(error);
    res.status(500).json({ reply: `Nexora AI is having trouble processing your request right now. Details: ${errorMessage}` });
  }
});

// Vite middleware for development
async function startServer() {
  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), 'dist');
    app.use(express.static(distPath));
    app.get('*', (req, res) => {
      res.sendFile(path.join(distPath, 'index.html'));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`Nexora AI running on http://localhost:${PORT}`);
  });
}

startServer();
