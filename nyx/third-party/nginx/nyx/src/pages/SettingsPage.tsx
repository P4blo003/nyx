// ==========================================================================================
// Author: Pablo González García.
// Created: 19/02/2026
// Last edited: 20/02/2026
// ==========================================================================================


// ==============================
// IMPORTS
// ==============================

// Internal:
import SettingsSidebar from "../components/settings/SettingsSidebar";


// ==============================
// COMPONENTS
// ==============================

const SettingsPage = () => {

  return (
    <div className="flex h-full overflow-hidden">
      {/* Sidebar — always visible on md+, hidden on mobile when a chat is selected */}
      <div className={`w-full md:w-80 md:block shrink-0`}>
          <SettingsSidebar/>
      </div>
    </div>
  );
}


// ==============================
// EXPORTS
// ==============================

export default SettingsPage