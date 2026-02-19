import { SearchIcon } from "lucide-react";
import TextInput from "./TextInput";


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

export default SearchInput;