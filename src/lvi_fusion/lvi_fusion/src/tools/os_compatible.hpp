#ifndef __OS_COMPATIBLE_HPP__
#define __OS_COMPATIBLE_HPP__
#include <string>
#include <fstream>
#include <iostream>
#include <ostream>
#include <sstream>
#include <stdarg.h>     //need for such like printf(...)
#include <stdio.h>
#include <string>
// #define  __GNUC__  1
#if defined _MSC_VER
#include <direct.h>
#elif defined __GNUC__
#include <sys/types.h>
#include <sys/stat.h>
#endif
 #include <pwd.h>
using namespace std;

#define PATH_SEPARATOR '/'
#define PATH_SEPARATOR_STR "/"
#define REVERSE_PATH_SEPARATOR '\\'

namespace Common_tools
{
    static std::string &trimUnifySlash( std::string &path )
    {
        std::string::size_type start = 1;
        while ( ( start = path.find( PATH_SEPARATOR, start ) ) != std::string::npos )
            if ( path[ start - 1 ] == PATH_SEPARATOR )
                path.erase( start, 1 );
            else
                ++start;
        return path;
    }

    static std::string &ensureUnifySlash( std::string &path )
    {
        std::string::size_type start = 0;
        while ( ( start = path.find( REVERSE_PATH_SEPARATOR, start ) ) != std::string::npos )
            path[ start ] = PATH_SEPARATOR;
        return trimUnifySlash( path );
    }

    inline bool if_file_exist( const std::string &name )
    {
        // Copy from: https://stackoverflow.com/questions/12774207/fastest-way-to-check-if-a-file-exist-using-standard-c-c11-c
        struct stat buffer;
        return ( stat( name.c_str(), &buffer ) == 0 );
    }

    inline std::string get_home_folder()
    {
    #ifdef _MSC_VER
        TCHAR homedir[ MAX_PATH ];
        if ( SHGetSpecialFolderPath( 0, homedir, CSIDL_PROFILE, TRUE ) != TRUE )
            return std::string();
    #else
        const char *homedir;
        if ( ( homedir = getenv( "HOME" ) ) == NULL )
            homedir = getpwuid( getuid() )->pw_dir;
    #endif // _MSC_VER
        std::string dir( std::string( homedir ) + PATH_SEPARATOR );
        return ensureUnifySlash( dir );
    }

    inline void create_dir(std::string dir)
    {
#if defined _MSC_VER
        _mkdir(dir.data());
#elif defined __GNUC__
        mkdir(dir.data(), 0777);
#endif
    }

    // Using asprintf() on windows
    // https://stackoverflow.com/questions/40159892/using-asprintf-on-windows
#ifndef _vscprintf
/* For some reason, MSVC fails to honour this #ifndef. */
/* Hence function renamed to _vscprintf_so(). */
    inline int _vscprintf_so(const char * format, va_list pargs) {
        int retval;
        va_list argcopy;
        va_copy(argcopy, pargs);
        retval = vsnprintf(NULL, 0, format, argcopy);
        va_end(argcopy);
        return retval;
    }
#endif // 

#ifndef vasprintf
    inline int vasprintf(char **strp, const char *fmt, va_list ap) {
        int len = _vscprintf_so(fmt, ap);
        if (len == -1) return -1;
        char *str = (char*)malloc((size_t)len + 1);
        if (!str) return -1;
        int r = vsnprintf(str, len + 1, fmt, ap); /* "secure" version of vsprintf */
        if (r == -1) return free(str), -1;
        *strp = str;
        return r;
    }
#endif // vasprintf
}
#endif