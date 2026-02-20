// ==========================================================================================
// Author: Pablo González García.
// Created: 19/02/2026
// Last edited: 20/02/2026
// ==========================================================================================


// ==============================
// IMPORTS
// ==============================

// Standard:
import { memo, useCallback, useEffect, useRef, useState } from "react";

// External:
import { EllipsisVertical, Pin, Pencil, Trash2 } from "lucide-react";

// Internal:
import type { Chat } from "../../store/chat/types";


// ==============================
// PROPERTIES
// ==============================

interface ChatListItemProps
{
    chat: Chat;
    isSelected: boolean;
    onSelect: (id:string) => void;
    onPin: (id:string) => void;
    onRename: (id:string, title:string) => void;
    onDelete: (id:string) => void;
}


// ==============================
// COMPONENTS
// ==============================

const ChatListItem = memo(({ chat, isSelected, onSelect, onPin, onRename, onDelete }:ChatListItemProps) =>
{
    const [isEditing, setIsEditing] = useState(false);
    const [editValue, setEditValue] = useState(chat.title);
    const inputRef = useRef<HTMLInputElement>(null);

    useEffect(() =>
    {
        if (isEditing)
            inputRef.current?.select();
    }, [isEditing]);

    const closeDropdown = () =>
    {
        if (document.activeElement instanceof HTMLElement)
            document.activeElement.blur();
    };

    const handleAction = (e:React.MouseEvent, action:() => void) =>
    {
        e.stopPropagation();
        closeDropdown();
        action();
    };

    const startEditing = () =>
    {
        setEditValue(chat.title);
        setIsEditing(true);
    };

    const confirmRename = useCallback(() =>
    {
        const trimmed = editValue.trim();

        if (trimmed && trimmed !== chat.title)
            onRename(chat.id, trimmed);

        setIsEditing(false);
    }, [editValue, chat.id, chat.title, onRename]);

    const cancelEditing = useCallback(() =>
    {
        setEditValue(chat.title);
        setIsEditing(false);
    }, [chat.title]);

    const handleKeyDown = (e:React.KeyboardEvent) =>
    {
        if (e.key === "Enter")
            confirmRename();
        else if (e.key === "Escape")
            cancelEditing();
    };

    return (
        <div
            onClick={() => !isEditing && onSelect(chat.id)}
            className={`group w-full text-left p-3 h-12 rounded-lg transition-colors duration-200 cursor-pointer flex items-center gap-2
                hover:bg-base-300 text-base-content ${isEditing || isSelected ? "bg-base-300" : ""}`}
        >
            <div className="flex-1 min-w-0">
                <input
                    ref={inputRef}
                    value={isEditing ? editValue : chat.title}
                    onChange={(e) => setEditValue(e.target.value)}
                    onBlur={isEditing ? confirmRename : undefined}
                    onKeyDown={isEditing ? handleKeyDown : undefined}
                    onClick={isEditing ? (e) => e.stopPropagation() : undefined}
                    readOnly={!isEditing}
                    className="font-medium w-full bg-transparent border-none outline-none h-6 p-0 m-0 leading-normal truncate cursor-pointer"
                />

                {chat.lastMessage && !isEditing && (
                    <p className="text-sm text-base-content/50 truncate mt-1">
                        {chat.lastMessage}
                    </p>
                )}
            </div>

            {/* Context menu */}
            {!isEditing && (
                <div className="dropdown dropdown-end" onClick={(e) => e.stopPropagation()}>
                    <div
                        tabIndex={0}
                        role="button"
                        className="btn btn-ghost btn-sm btn-square border-none shadow-none text-base-content/50 hover:text-base-content hover:scale-110 transition-all duration-200 shrink-0"
                    >
                        <EllipsisVertical size={18}/>
                    </div>

                    <ul tabIndex={0} className="dropdown-content menu bg-base-200 rounded-box w-48 shadow-lg z-10 p-2">
                        <li>
                            <button onClick={(e) => handleAction(e, () => onPin(chat.id))}>
                                <Pin size={16}/> Fijar chat
                            </button>
                        </li>
                        <li>
                            <button onClick={(e) => handleAction(e, startEditing)}>
                                <Pencil size={16}/> Editar nombre
                            </button>
                        </li>
                        <li>
                            <button
                                onClick={(e) => handleAction(e, () => onDelete(chat.id))}
                                className="text-red-400 hover:text-red-400!"
                            >
                                <Trash2 size={16}/> Eliminar
                            </button>
                        </li>
                    </ul>
                </div>
            )}
        </div>
    );
});


// ==============================
// EXPORTS
// ==============================

export default ChatListItem;