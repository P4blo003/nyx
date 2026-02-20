// ==========================================================================================
// Author: Pablo González García.
// Created: 19/02/2026
// Last edited: 20/02/2026
// ==========================================================================================


// ==============================
// IMPORTS
// ==============================

// Standard:
import { useCallback, useRef, useState } from "react";

// Internal:
import { Paperclip, Mic, SendHorizontal } from "lucide-react";


// ==============================
// PROPERTIES
// ==============================

interface ChatInputProps
{
    onSendMessage: (content:string) => void;
}


// ==============================
// COMPONENTS
// ==============================

const ChatInput = ({ onSendMessage }:ChatInputProps) =>
{
    const [text, setText] = useState("");
    const textareaRef = useRef<HTMLTextAreaElement>(null);
    const baseHeight = useRef<number>(0);

    const MAX_HEIGHT = 160;
    const hasText = text.trim().length > 0;

    const initRef = useCallback((node:HTMLTextAreaElement | null) =>
    {
        textareaRef.current = node;

        if (node && !baseHeight.current)
            baseHeight.current = node.scrollHeight;
    }, []);

    const adjustHeight = () =>
    {
        const ta = textareaRef.current;
        if (!ta) return;

        ta.style.height = "auto";

        const target = Math.max(ta.scrollHeight, baseHeight.current);
        const clamped = Math.min(target, MAX_HEIGHT);

        ta.style.height = `${clamped}px`;
        ta.style.overflowY = target > MAX_HEIGHT ? "auto" : "hidden";
    };

    const handleChange = (e:React.ChangeEvent<HTMLTextAreaElement>) =>
    {
        setText(e.target.value);
        adjustHeight();
    };

    const handleSend = () =>
    {
        const trimmed = text.trim();

        if (!trimmed) return;

        onSendMessage(trimmed);
        setText("");

        const ta = textareaRef.current;

        if (ta)
        {
            ta.style.height = `${baseHeight.current}px`;
            ta.style.overflowY = "hidden";
        }
    };

    const handleKeyDown = (e:React.KeyboardEvent) =>
    {
        if (e.key === "Enter" && !e.shiftKey)
        {
            e.preventDefault();
            handleSend();
        }
    };

    return (
        <div className="p-4 border-t border-base-content/10">
            <div className="flex items-end gap-2">
                <button className="btn btn-ghost btn-circle btn-sm mb-0.5">
                    <Paperclip size={18}/>
                </button>

                <textarea
                    ref={initRef}
                    value={text}
                    onChange={handleChange}
                    onKeyDown={handleKeyDown}
                    placeholder="Escribe un mensaje..."
                    rows={1}
                    className="textarea flex-1 rounded-3xl text-base bg-base-200 border-base-content/20 focus:outline-none focus:ring-0 focus:border-primary placeholder:text-base-content/40 resize-none leading-6 py-2 px-4 min-h-0 overflow-y-hidden"
                />

                <button
                    className={`btn btn-circle btn-sm mb-0.5 ${hasText ? "btn-primary" : "btn-ghost"}`}
                    onClick={hasText ? handleSend : undefined}
                >
                    {hasText ? <SendHorizontal size={18}/> : <Mic size={18}/>}
                </button>
            </div>
        </div>
    );
};


// ==============================
// EXPORTS
// ==============================

export default ChatInput;