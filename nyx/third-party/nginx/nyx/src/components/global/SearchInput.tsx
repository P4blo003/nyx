// ==========================================================================================
// Author: Pablo González García.
// Created: 19/02/2026
// Last edited: 20/02/2026
// ==========================================================================================


// ==============================
// IMPORTS
// ==============================

// External:
import { SearchIcon } from "lucide-react";

// Internal:
import TextInput from "./TextInput";


// ==============================
// COMPONENTS
// ==============================

const SearchInput = () =>
{
    return (
        <div className="mt-3 relative">
            <TextInput
                type="text"
                icon={<SearchIcon size={16}/>}
                placeholder="Buscar ..."
            />
        </div>
    )
}


// ==============================
// EXPORTS
// ==============================

export default SearchInput;