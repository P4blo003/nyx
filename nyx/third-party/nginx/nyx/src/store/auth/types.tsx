// ==========================================================================================
// Author: Pablo González García.
// Created: 19/02/2026
// Last edited: 20/02/2026
// ==========================================================================================


// ==============================
// INTERFACES
// ==============================

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