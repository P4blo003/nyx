// ==========================================================================================
// Author: Pablo González García.
// Created: 19/02/2026
// Last edited: 20/02/2026
// ==========================================================================================


// ==============================
// IMPORTS
// ==============================

// External:
import Keycloak from "keycloak-js";


// ==============================
// INSTANCES
// ==============================

const keycloak = new Keycloak({
    url: "http://localhost:8080",
    realm: "assistant",
    clientId: "frontend"
});


// ==============================
// METHODS
// ==============================

export function getTokenMinValidity():number
{
    const exp = keycloak.tokenParsed?.exp;
    const iat = keycloak.tokenParsed?.iat;

    if (!exp || !iat) return 10;

    return Math.max(Math.floor((exp - iat) * 0.1), 5);
}


// ==============================
// EXPORTS
// ==============================

export default keycloak;