


export interface AuthUser
{
    // ---- Properties ---- //

    fullname: string,
    email: string,
    token: string
}


export interface AuthState
{
    // ---- Properties ---- //

    authUser: AuthUser | null;

    isInitializing: boolean;
    isConnectionError: boolean;


    // ---- Methods ---- //

    initKeycloak: () => void;
    login: () => void;
    logout: () => void;
}
