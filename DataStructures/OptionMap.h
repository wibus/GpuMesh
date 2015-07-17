#ifndef GPUMESH_OPTIONMAP
#define GPUMESH_OPTIONMAP

#include <map>
#include <string>
#include <sstream>

#include <CellarWorkbench/Misc/Log.h>


struct OptionMapDetails
{
    std::string title;
    std::string defaultOption;
    std::vector<std::string> options;
};

template<typename V>
class OptionMap
{
public:
    typedef std::map<std::string, V> Content;

    OptionMap(const std::string& title);
    OptionMap(const std::string& title,
              const std::string& defaultOptionKey,
              const Content& contentMap);

    void setDefault(const std::string& defaultOptionKey);

    void setContent(const Content& contentMap);

    bool select(const std::string& key, V& value) const;

    OptionMapDetails details() const;

private:
    std::string _title;
    std::string _defaultOption;
    Content _content;
};



// IMPLEMENTATION //
template<typename V>
OptionMap<V>::OptionMap(const std::string& title) :
    _title(title),
    _defaultOption(),
    _content()
{

}

template<typename V>
OptionMap<V>::OptionMap(const std::string& title,
                           const std::string& defaultOptionKey,
                           const Content& contentMap) :
    _title(title),
    _defaultOption(defaultOptionKey),
    _content(contentMap)
{

}

template<typename V>
void OptionMap<V>::setDefault(const std::string& defaultOptionKey)
{
    _defaultOption = defaultOptionKey;
}

template<typename V>
void OptionMap<V>::setContent(const Content& contentMap)
{
    _content = contentMap;
}

template<typename V>
bool OptionMap<V>::select(const std::string& key, V& value) const
{
    auto it = _content.find(key);
    if(it != _content.end())
    {
        value = it->second;
        return true;
    }
    else
    {
        std::stringstream ss; ss << key;
        cellar::getLog().postMessage(new cellar::Message('E', false,
            "'" + ss.str() + "' is not a valid option for '" + _title + "'",
            "OptionMap"));
        return false;
    }
}

template<typename V>
OptionMapDetails OptionMap<V>::details() const
{
    OptionMapDetails det;
    det.title = _title;
    det.defaultOption = _defaultOption;
    for(const auto& keyValue : _content)
        det.options.push_back(keyValue.first);

    return det;
}

#endif // GPUMESH_OPTIONMAP
