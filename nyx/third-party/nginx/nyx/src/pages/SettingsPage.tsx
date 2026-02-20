import SettingsSidebar from "../components/settings/SettingsSidebar";

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

export default SettingsPage