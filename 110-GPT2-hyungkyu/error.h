#ifndef ERROR_H
#define ERROR_H

#include <source_location>
#include <cstdlib>
#include <cstdio>


template<typename T>
inline void assert_impl(
    const T& condition,
    const char* condition_str,
    const std::source_location location = std::source_location::current()
) {
#ifndef NDEBUG
    if (!static_cast<bool>(condition)) {
        fprintf(stderr,
            "[ASSERT FAILED] \"%s\" at %s in %s(%d:%d)\n",
            condition_str,
            location.function_name(),
            location.file_name(),
            static_cast<int>(location.line()),
            static_cast<int>(location.column())
        );
        std::abort();
    }
#endif
}

#define ASSERT_(expr) assert_impl((expr), #expr, std::source_location::current())

// Undefine Windows SDK's _ASSERT to avoid macro redefinition warning
#ifdef _ASSERT
#undef _ASSERT
#endif
#define _ASSERT(expr) ASSERT_(expr)  // Alias for compatibility


#endif // ERROR_H

