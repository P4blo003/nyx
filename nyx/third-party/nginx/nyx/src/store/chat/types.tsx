

// ==============================
// Entities
// ==============================

export interface Chat
{
    id: string;
    title: string;
    lastMessage: string;
    updatedAt: string;
    isPinned: boolean;
}


// ==============================
// State
// ==============================

export interface ChatState
{
    // ---- Properties ---- //

    chats: Chat[];
    selectedChatId: string | null;


    // ---- Methods ---- //

    selectChat: (id: string) => void;
    createChat: () => void;
    deleteChat: (id: string) => void;
    pinChat: (id: string) => void;
    renameChat: (id: string, title: string) => void;
}
