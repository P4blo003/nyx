
// External:
import { Route, Routes } from "react-router-dom";

// Internal:
import Navbar from "./components/Navbar";
import HomePage from "./pages/HomePage";
import SettingsPage from "./pages/SettingsPage";
import ProfilePage from "./pages/ProfilePage";
import ConnectionErrorPage from "./pages/ConnectionErrorPage";
import { useAuthStore } from "./store/auth/useAuthStore";
import { useEffect } from "react";
import { Loader } from "lucide-react";

const App = () =>
{
  const {authUser, initKeycloak, isInitializing, isConnectionError} = useAuthStore();

  useEffect(() => {initKeycloak();}, [initKeycloak])

  if (isInitializing && !authUser)
  {
    return (
      <div className="flex items-center justify-center h-screen">
        <Loader className="size-10 animate-spin"/>
      </div>
    )
  }

  if (isConnectionError)
    return <ConnectionErrorPage />;

  return (
    <div>
      <Navbar />

      <Routes>
        <Route path="/" element={<HomePage/>}/>
        <Route path="/settings" element={<SettingsPage/>}/>
        <Route path="/profile" element={<ProfilePage/>}/>
      </Routes>
    </div>
  )
}


export default App;
