import ProfileSidebar from "../components/profile/ProfileSidebar";

const ProfilePage = () => {

  return (
    <div className="flex h-full overflow-hidden">
      {/* Sidebar — always visible on md+, hidden on mobile when a chat is selected */}
      <div className={`w-full md:w-80 md:block shrink-0`}>
          <ProfileSidebar/>
      </div>
    </div>
  );
};

export default ProfilePage;