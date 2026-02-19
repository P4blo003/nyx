import ChatSidebar from "../components/chat/ChatSidebar";


// ==============================
// Component
// ==============================

const HomePage = () =>
{
    return (
        <div className="flex h-screen overflow-hidden">

            {/* Sidebar — always visible on md+, hidden on mobile when a chat is selected */}
            <div className={`w-full md:w-80 md:block shrink-0`}>
                <ChatSidebar/>
            </div>
        </div>
    );
};

export default HomePage;
