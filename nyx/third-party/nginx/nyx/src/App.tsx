// ==========================================================================================
// Author: Pablo González García.
// Created: 19/02/2026
// Last edited: 20/02/2026
// ==========================================================================================


// ==============================
// IMPORTS
// ==============================

// Internal:
import { useEffect } from "react";

// External:
import { Route, Routes } from "react-router-dom";
import { Loader } from "lucide-react";

// Internal:
import Navbar from "./components/global/Navbar";
import HomePage from "./pages/HomePage";
import SettingsPage from "./pages/SettingsPage";
import ProfilePage from "./pages/ProfilePage";
import ConnectionErrorPage from "./pages/ConnectionErrorPage";
import LandingPage from "./pages/LandingPage";
import { useAuthStore } from "./store/auth/useAuthStore";
import DocumentsPage from "./pages/DocumentsPage";


// ==============================
// COMPONENTS
// ==============================

const App = () =>
{
  const {authUser, initKeycloak, isInitializing, isConnectionError} = useAuthStore();
  
  useEffect(() => {initKeycloak();}, [initKeycloak])

  if (isInitializing)
  {
    return (
      <div className="flex items-center justify-center h-screen">
        <Loader className="size-10 animate-spin"/>
      </div>
    )
  }

  if (isConnectionError)
    return <ConnectionErrorPage />;

  if (!authUser)
    return <LandingPage />;

  return (
    <div className="flex flex-col-reverse md:flex-row h-screen">
      <Navbar username={authUser.fullname}/>
      <div className="flex-1 flex flex-col min-h-0 min-w-0">
        <Routes>
          <Route path="/" element={<HomePage/>}/>
          <Route path="/documents" element={<DocumentsPage/>}/>
          <Route path="/settings" element={<SettingsPage/>}/>
          <Route path="/profile" element={<ProfilePage/>}/>
        </Routes>
      </div>
    </div>
  )
}


// ==============================
// EXPORTS
// ==============================

export default App;