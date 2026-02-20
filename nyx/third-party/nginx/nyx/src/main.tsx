// ==========================================================================================
// Author: Pablo González García.
// Created: 19/02/2026
// Last edited: 20/02/2026
// ==========================================================================================


// ==============================
// IMPORTS
// ==============================

// Standard:
import { StrictMode } from 'react'

// External:
import { createRoot } from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'

// Internal:
import './index.css'
import App from './App.tsx'


// ==============================
// MAIN
// ==============================

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <BrowserRouter>
      <App />
    </BrowserRouter>
  </StrictMode>,
)
