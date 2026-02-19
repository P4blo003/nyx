import type { ReactNode } from "react";
import type React from "react";



interface TextInputProps
{
    type:string
    value?:string,
    onChange?: (e:React.ChangeEvent<HTMLInputElement>) => void,
    placeholder?:string,
    icon?: ReactNode
}

const TextInput = ({type, value, onChange, placeholder, icon}:TextInputProps) =>
{
    return (
        <div className="relative w-full">
            {icon && (
                <div className="absolute inset-y-0 left-3 flex items-center text-base-content/40 z-10">
                    {icon}
                </div>
            )}
            <input 
                type={type ?? "text"}
                value={value}
                onChange={onChange}
                placeholder={placeholder}
                className={
                    `input input-bordered w-full h-10 ${icon ? "pl-10" : "pl-4"}
                    rounded-full text-base bg-base-200 border-base-content/20
                    focus:outline-none focus:ring-0 focus:border-primary
                    placeholder:text-base-content/40    
                `}
            />
        </div>
    )
}

export default TextInput